import asyncio
import functools
import logging
import os
from typing import List

import itertools
import pytest

import ubii.proto as ub
from ubii.framework import processing
from ubii.framework.client import InitProcessingModules, RunProcessingModules, Publish, Subscriptions
from ubii.processing_modules.ocr.tesseract_ocr import (
    TesseractOCR_MSER,
    TesseractOCR_EAST,
    TesseractOCR_PURE
)

try:
    from PIL import ImageDraw
except ImportError:
    import ImageDraw

try:
    from PIL import Image
except ImportError:
    import Image

try:
    from PIL import ImageFont
except ImportError:
    import ImageFont

log = logging.getLogger(__name__)

PIL_to_proto_format = {
    'RGBA': ub.Image2D.DataFormat.RGBA8,
    'RGB': ub.Image2D.DataFormat.RGB8
}

proto_format_to_PIL = {v: k for k, v in PIL_to_proto_format.items()}


@pytest.fixture(autouse=True, scope='session')
def debug_settings():
    """
    Don't turn on debug settings
    """
    from ubii.framework import debug
    debug(enabled=False)
    yield


@pytest.fixture(scope='session')
def image_data(data_dir, request):
    data = Image.open(data_dir / request.param)
    log.info(f'Loaded image {data}')

    try:
        image_fmt = PIL_to_proto_format[data.mode]
    except KeyError as e:
        raise ValueError(f"Format {data.mode} not supported") from e

    yield ub.Image2D(
        width=data.width,
        height=data.height,
        data_format=image_fmt,
        data=data.tobytes()
    )


@pytest.fixture(scope='session')
def font(data_dir, request):
    fnt = ImageFont.truetype(os.fspath(data_dir / request.param), 40)
    yield fnt


class TestOCRPerformance:
    client_spec = [
        pytest.param((ub.Client(is_dedicated_processing_node=True, ),), id='processing'),
    ]

    late_init_module_spec = itertools.chain.from_iterable(
        (
            pytest.param((functools.partial(v, filter_empty_boxes=True),),
                         id=f"{v!r}-empty"),
            pytest.param((functools.partial(v, filter_empty_boxes=False),),
                         id=f"{v!r}"),
            pytest.param((functools.partial(v, filter_empty_boxes=True, ocr_confidence=50),),
                         id=f"{v!r}-empty-50"),
        )
        for v in (
            TesseractOCR_PURE,
            functools.partial(
                TesseractOCR_EAST,
                nms_threshold=0.5,
            ),
            functools.partial(
                TesseractOCR_EAST,
                merge_bounding_boxes=True,
                nms_threshold=0.7,
            ),
            functools.partial(
                TesseractOCR_EAST,
                nms_threshold=0.7,
            ),
            TesseractOCR_MSER,
        )
    )

    @pytest.fixture(scope='class')
    async def base_session(self, client, start_client) -> ub.Session:
        await start_client(client)
        module = client.processing_modules[0]
        input_topic = f"{client.id}/test_input"
        output_topic = f"{client.id}/test_output"
        io_mappings = [
            {
                'processing_module_name': module.name,
                'input_mappings': [
                    {
                        'input_name': module.inputs[0].internal_name,
                        'topic': input_topic
                    },
                ],
                'output_mappings': [
                    {
                        'output_name': module.outputs[0].internal_name,
                        'topic': output_topic
                    }
                ]
            },
        ]

        session = ub.Session(name=f"session-{client.id}",
                             processing_modules=[module],
                             io_mappings=io_mappings)

        return session

    @pytest.fixture(autouse=True)
    async def startup(self, client, start_client, stop_client, session_spec, start_session):
        await start_client(client)

        await client.implements(InitProcessingModules, RunProcessingModules)
        await start_session(session_spec)

        pm: processing.ProcessingRoutine = processing.ProcessingRoutine.registry[
            session_spec.processing_modules[0].name
        ]
        async with pm.change_specs:
            await pm.change_specs.wait_for(
                lambda: pm.status == pm.Status.CREATED or pm.status == pm.Status.PROCESSING
            )
        yield
        await stop_client(client)

    @pytest.mark.parametrize('image_data', [
        # pytest.param('test.png', id='testbild'),
        pytest.param('Ortseingangsschild_Ilmenau.jpg', id='ortsschild')
    ], indirect=True)
    @pytest.mark.parametrize('timeout', [1])
    @pytest.mark.parametrize('run_time', [4])
    @pytest.mark.parametrize('font', [pytest.param('TeX_Gyre_Heros_Regular.otf', id='heros')], indirect=True)
    async def test_processing_module(
            self,
            client,
            image_data: ub.Image2D,
            base_session: ub.Session,
            timeout,
            run_time,
            data_dir,
            font,
            request,
    ):
        received: List[ub.TopicDataRecord] = []
        input_topic = base_session.io_mappings[0].input_mappings[0].topic
        output_topic = base_session.io_mappings[0].output_mappings[0].topic

        await client[Publish].publish({
            'topic': input_topic,
            'image2D': image_data
        })

        await asyncio.sleep(timeout)  # we wait until the broker processed the publishing
        topics, tokens = await client[Subscriptions].subscribe_topic(output_topic).with_callback(received.append)
        # if you subscribe to the same topic as before, you will immediately get the last value, therefore
        # we publish before we subscribe!

        await asyncio.sleep(run_time)
        await topics[0].unregister_callback(tokens[0], timeout=timeout)
        await client[Subscriptions].unsubscribe_topic(output_topic)

        assert received, "No processed images received"
        # assert all(received[0] == r for r in received)

        boxes = received[-1].object2D_list
        used = Image.frombuffer(
            proto_format_to_PIL[image_data.data_format],
            (image_data.width, image_data.height),
            image_data.data
        )
        draw = ImageDraw.Draw(used)

        for box in boxes.elements:
            x, y = box.pose.position.x * image_data.width, box.pose.position.y * image_data.height
            w, h = box.size.x * image_data.width, box.size.y * image_data.height

            draw.rectangle(((x, y), (x + w, y + h)), outline=(0, 255, 0))
            draw.text((x + w, y), text=box.id, font=font, anchor='rb', fill=(0, 255, 0))

        used.save(data_dir / 'results' / f"{request.node.callspec.id}.png")
