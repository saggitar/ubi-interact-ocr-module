import asyncio
import logging
import os
import warnings
from typing import List

import pytest

import ubii.proto as ub
from ubii.framework import processing
from ubii.framework.client import InitProcessingModules, RunProcessingModules, Publish, Subscriptions
from ubii.processing_modules.ocr.tesseract_ocr import TesseractOCR_PURE, TesseractOCR_MSER, TesseractOCR_EAST

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


@pytest.fixture(scope='session')
def image_data(data_dir, request):
    format_map = {
        'RGBA': ub.Image2D.DataFormat.RGBA8
    }

    data = Image.open(data_dir / request.param)
    log.info(f'Loaded image {data}')

    try:
        format = format_map[data.mode]
    except KeyError as e:
        raise ValueError(f"Format {data.mode} not supported") from e

    yield ub.Image2D(
        width=data.width,
        height=data.height,
        data_format=format,
        data=data.tobytes()
    )


@pytest.fixture(scope='session')
def font(data_dir):
    fnt = ImageFont.truetype(os.fspath(data_dir / "TeX_Gyre_Heros_Regular.otf"), 40)
    yield fnt


class TestOCRPerformance:
    client_spec = [
        pytest.param((ub.Client(
            is_dedicated_processing_node=True,
        ),), id='processing_node')
    ]

    late_init_module_spec = [
        pytest.param((TesseractOCR_EAST,), id='EAST'),
        pytest.param((TesseractOCR_MSER,), id='MSER'),
        pytest.param((TesseractOCR_PURE,), id='PURE'),
    ]

    @pytest.fixture(scope='class')
    async def base_session(self, client) -> ub.Session:
        try:
            await client
        except UserWarning as w:
            log.info(w)
            client.protocol.start()

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='ubii.framework.client')
                await client

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
    async def startup(self, client, session_spec, start_session):
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
        await client.protocol.stop()

    @pytest.mark.parametrize('image_data', ['test.png'], indirect=True)
    @pytest.mark.parametrize('timeout', [0.4])
    @pytest.mark.parametrize('run_time', [4])
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

        tested_module: ub.ProcessingModule = client[InitProcessingModules].initialized[0]
        assert all(received[0] == r for r in received)

        boxes = received[0].object2D_list
        used = Image.frombuffer('RGBA', (image_data.width, image_data.height), image_data.data)
        draw = ImageDraw.Draw(used)

        for box in boxes.elements:
            x, y = box.pose.position.x * image_data.width, box.pose.position.y * image_data.height
            w, h = box.size.x * image_data.width, box.size.y * image_data.height

            draw.rectangle(((x, y), (x + w, y + h)), outline=(0,255,0))
            draw.text((x,y), text=box.id, font=font, anchor='rt', fill=(0,255,0))

        used.save(data_dir / f"{tested_module.name}-{request.node.callspec.params['image_data']}")
