import asyncio
import functools
import json
import logging
import os

import itertools
import pandas as pd
import pytest
from tesserocr import OEM, PSM

import ubii.proto as ub
from ubii.framework.client import InitProcessingModules, RunProcessingModules, Publish, Subscriptions, Sessions
from ubii.processing_modules.ocr.tesseract_ocr import (
    TesseractOCR_PURE,
    TesseractOCR_EAST,
    TesseractOCR_MSER
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
def debug_settings(configure_logging):
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
    fnt = ImageFont.truetype(os.fspath(data_dir / getattr(request, 'param', 'TeX_Gyre_Heros_Regular.otf')), 40)
    yield fnt


@pytest.fixture(scope='session')
def run_time(data_dir, request):
    return getattr(request, 'param', 20)


class TestOCRPerformance:

    @pytest.fixture(scope='class')
    async def base_session(self, client) -> ub.Session:
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

        session = ub.Session(name='OCR Session',
                             processing_modules=[module],
                             io_mappings=io_mappings)

        yield session

    client_spec = [
        pytest.param((ub.Client(is_dedicated_processing_node=True, ),), id='pm'),
    ]

    late_init_module_spec = list(itertools.chain.from_iterable(
        (
            pytest.param(functools.partial(paramset.values[0], filter_empty_boxes=True),
                         id=f"{paramset.id}-empty"),
            pytest.param(functools.partial(paramset.values[0], filter_empty_boxes=False),
                         id=f"{paramset.id}"),
            pytest.param(functools.partial(paramset.values[0], filter_empty_boxes=False, ocr_confidence=20),
                         id=f"{paramset.id}-20"),
            pytest.param(functools.partial(paramset.values[0], filter_empty_boxes=True, ocr_confidence=50),
                         id=f"{paramset.id}-empty-50"),
        )
        for paramset in (
            # pytest.param(
            #     TesseractOCR_PURE,
            #     id='PURE'
            # ),
            # pytest.param(
            #     functools.partial(
            #         TesseractOCR_PURE,
            #         api_args={'oem': OEM.DEFAULT, 'psm': PSM.SPARSE_TEXT_OSD}
            #     ),
            #     id='PURE-sparse'
            # ),
            # pytest.param(
            #     TesseractOCR_MSER,
            #     id='MSER'
            # ),
            # pytest.param(
            #     functools.partial(
            #         TesseractOCR_MSER,
            #         padding=0,
            #     ),
            #     id='MSER-padding0'
            # ),
            # pytest.param(
            #     functools.partial(
            #         TesseractOCR_MSER,
            #         padding=3,
            #         mser_args={'max_variation': 0.02},
            #     ),
            #     id='MSER-padding3-variation0.02'
            # ),
            # pytest.param(
            #     functools.partial(
            #         TesseractOCR_MSER,
            #         padding=3,
            #     ),
            #     id='MSER-padding3'
            # ),
            pytest.param(
                functools.partial(
                    TesseractOCR_MSER,
                    mser_args={'max_variation': 0.1}
                ),
                id='MSER-variation0.1'
            ),
            pytest.param(
                functools.partial(
                    TesseractOCR_EAST,
                    nms_threshold=0.5,
                ),
                id='EAST-nms0.5-no-merge'
            ),
            pytest.param(
                functools.partial(
                    TesseractOCR_EAST,
                    merge_bounding_boxes=True,
                    nms_threshold=0.7,
                ),
                id='EAST-nms0.7-merge'
            ),
            pytest.param(
                functools.partial(
                    TesseractOCR_EAST,
                    nms_threshold=0.7,
                ),
                id='EAST-nms0.7-no-merge'
            ),
        )
    ))

    @pytest.mark.parametrize('image_data', [
        pytest.param('test.png', id='test'),
        pytest.param('webcam.png', id='webcam'),
        pytest.param('Ortseingangsschild_Ilmenau.jpg', id='ortsschild')
    ], indirect=True)
    async def test_processing_module(
            self,
            client,
            session_spec,
            image_data,
            request,
            caplog,
            data_dir,
            font,
            event_loop,
            reset_and_start_client,
            run_time,
            test_data,
    ):
        """
        Actually starting and stopping the session should be done in a fixture, somehow the pycharm test runner
        does not respect fixture order (or maybe it starts new tests when old tests tear down?) So that stopping
        clients and sessions during teardown can not be consistently ordered. Therefor we do it in the test, which
        sucks because when the test fails, the session is not stopped which could lead to unexpected behaviour.
        Still, it is better this way, since I can't get the fixture order working when not in debug mode.

        If tests fail, and you see memory leaks due to residual sessions in the broker, stop the test suite before
        your system is clogged down!
        """
        caplog.set_level(logging.WARNING)
        await client.implements(InitProcessingModules, RunProcessingModules, Sessions)

        received_data = []
        input_topic = session_spec.io_mappings[0].input_mappings[0].topic
        output_topic = session_spec.io_mappings[0].output_mappings[0].topic
        module_name = session_spec.processing_modules[0].name

        started = await client[Sessions].start_session(session_spec)
        pm = await client[RunProcessingModules].get_module_instance(module_name)

        # first write performance data even for failed tests, disable by setting
        # write_test_references=False in pytest.ini
        interesting_values = {
            'hz': pm.processing_mode.frequency.hertz,
            'args': getattr(request.node.callspec.params['late_init_module_spec'], 'keywords', {}),
        }

        topics, tokens = await client[Subscriptions].subscribe_topic(output_topic).with_callback(received_data.append)
        await client[Publish].publish({
            'topic': input_topic,
            'image2D': image_data
        })

        start_time = event_loop.time()
        while len(received_data) <= 50 and event_loop.time() - start_time < run_time:
            await asyncio.sleep(1. / interesting_values['hz'])

        await topics[0].unregister_callback(tokens[0])
        await client[Sessions].stop_session(started)

        boxes = received_data[-1].object2D_list if received_data else []
        used = Image.frombuffer(
            proto_format_to_PIL[image_data.data_format],
            (image_data.width, image_data.height),
            image_data.data
        )
        draw = ImageDraw.Draw(used)

        for box in (boxes.elements if boxes else ()):
            x, y = box.pose.position.x * image_data.width, box.pose.position.y * image_data.height
            w, h = box.size.x * image_data.width, box.size.y * image_data.height

            draw.rectangle(((x, y), (x + w, y + h)), outline=(0, 255, 0), width=2)
            draw.text((x + w, y), text=box.id.replace('\n', '|'), font=font, anchor='rb', fill=(0, 255, 0))

        execution_times = pd.Series(pm._performance_values[1:])  # ignore first value where module started up

        with test_data.write(f"stats.txt") as f:
            statistics = execution_times.describe().to_frame().T
            statistics['relative std'] = statistics['std'] / statistics['mean']
            statistics.to_csv(f, index=False, float_format='%.4f')

        used.save(test_data.path("image.png"))

        with test_data.write(f"arguments.json") as f:
            json.dump(interesting_values, f)
