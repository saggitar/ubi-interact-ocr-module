import asyncio
import functools
import json
import logging
import os

import gc
import pandas as pd
import pytest
from tesserocr import OEM, PSM

import ubii.proto as ub
from ubii.framework.client import InitProcessingModules, RunProcessingModules, Publish, Subscriptions, Sessions
from ubii.processing_modules.ocr.tesseract_ocr import (
    TesseractOCR_EAST,
    TesseractOCR_MSER,
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
    MODULE_NAME = 'OCR Module'

    MSER_SPECS = [
        pytest.param(
            TesseractOCR_MSER,
            id='MSER'
        ),
        pytest.param(
            functools.partial(
                TesseractOCR_MSER,
                padding=0,
            ),
            id='MSER-padding0'
        ),
        pytest.param(
            functools.partial(
                TesseractOCR_MSER,
                padding=3,
                mser_args={'max_variation': 0.02},
            ),
            id='MSER-padding3-variation0_02'
        ),
        pytest.param(
            functools.partial(
                TesseractOCR_MSER,
                padding=3,
            ),
            id='MSER-padding3'
        ),
        pytest.param(
            functools.partial(
                TesseractOCR_MSER,
                padding=3,
                mser_args={'max_variation': 0.1},
                api_variables={'tessedit_char_whitelist': "ABCDFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"},
            ),
            id='MSER-variation0_1-charwhitelist'
        ),
    ]

    PURE_SPECS = [
        pytest.param(
            TesseractOCR_PURE,
            id='PURE'
        ),
        pytest.param(
            functools.partial(
                TesseractOCR_PURE,
                api_args={'oem': OEM.DEFAULT, 'psm': PSM.SPARSE_TEXT_OSD}
            ),
            id='PURE-sparse'
        ),
    ]

    EAST_SPECS = [
        pytest.param(
            functools.partial(
                TesseractOCR_EAST,
                nms_threshold=0.5,
            ),
            id='EAST-nms0_5-no-merge'
        ),
        pytest.param(
            functools.partial(
                TesseractOCR_EAST,
                merge_bounding_boxes=True,
                nms_threshold=0.6,
            ),
            id='EAST-nms0_6-merge'
        ),
        pytest.param(
            functools.partial(
                TesseractOCR_EAST,
                merge_bounding_boxes=False,
                nms_threshold=0.6,
            ),
            id='EAST-nms0_6-no-merge'
        ),
        pytest.param(
            functools.partial(
                TesseractOCR_EAST,
                merge_bounding_boxes=False,
                nms_threshold=0.7,
            ),
            id='EAST-nms0_7-no-merge'
        ),
    ]

    client_spec = [
        pytest.param(({'is_dedicated_processing_node': True, 'processing_modules': [{'name': MODULE_NAME}]},), id='pm'),
    ]

    @pytest.fixture
    async def session(self, ocr_client) -> ub.Session:
        input_topic = f"{ocr_client.id}/test_input"
        output_topic = f"{ocr_client.id}/test_output"
        module = ocr_client.processing_modules[0]

        io_mappings = [
            {
                'processing_module_name': module.name,
                'input_mappings': [
                    {
                        'input_name': 'image',
                        'topic': input_topic
                    },
                ],
                'output_mappings': [
                    {
                        'output_name': 'predictions',
                        'topic': output_topic
                    }
                ]
            },
        ]

        session = ub.Session(name='OCR Session',
                             processing_modules=[module],
                             io_mappings=io_mappings)

        await ocr_client.implements(Sessions)
        started = await ocr_client[Sessions].start_session(session)
        yield started
        await ocr_client[Sessions].stop_session(started)

    @pytest.fixture
    async def ocr_client(self, client, base_type, module_args, test_data):
        assert len(client.processing_modules) == 1
        pm_name = client.processing_modules[0].name
        factory = functools.partial(base_type, **module_args)
        client[InitProcessingModules].module_factories = {pm_name: factory}
        await client
        yield client
        await client.protocol.stop()

        interesting_values = {
            'hz': client.processing_modules[0].processing_mode.frequency.hertz,
            'args': getattr(factory, 'keywords', {}),
        }
        with test_data.write(f"arguments.json") as f:
            json.dump(interesting_values, f)

        await client.reset()
        gc.collect()

    @pytest.fixture
    async def ocr_module(self, ocr_client, session, test_data):

        await ocr_client.implements(RunProcessingModules)
        pm_name = session.processing_modules[0].name
        pm = await ocr_client[RunProcessingModules].get_module_instance(pm_name)
        yield pm

        # first write performance data even for failed tests, disable by setting
        # write_test_references=False in pytest.ini
        execution_times = pd.Series(pm._performance_values[1:])  # ignore first value where module started up

        with test_data.write(f"stats.txt") as f:
            statistics = execution_times.describe().to_frame().T
            statistics['relative std'] = statistics['std'] / statistics['mean']
            statistics.to_csv(f, index=False, float_format='%.4f')

    @pytest.fixture
    async def topic_data(self, ocr_client, session, image_data, test_data, font):

        received_data = []
        input_topic = session.io_mappings[0].input_mappings[0].topic
        output_topic = session.io_mappings[0].output_mappings[0].topic

        topics, tokens = await ocr_client[Subscriptions].subscribe_topic(output_topic).with_callback(
            received_data.append)

        await ocr_client[Publish].publish({
            'topic': input_topic,
            'image2D': image_data
        })

        yield received_data

        await topics[0].unregister_callback(tokens[0])

        boxes: ub.Object2DList = received_data[-1].object2D_list if received_data else []
        used = Image.frombuffer(
            proto_format_to_PIL[image_data.data_format],
            (image_data.width, image_data.height),
            image_data.data
        )
        draw = ImageDraw.Draw(used)
        words = []
        color = (255, 50, 255)

        for box in (boxes.elements if boxes else ()):
            x, y = box.pose.position.x * image_data.width, box.pose.position.y * image_data.height
            w, h = box.size.x * image_data.width, box.size.y * image_data.height

            draw.rectangle(((x, y), (x + w, y + h)), outline=color, width=2)
            draw.text((x + w, y), text=box.id.replace('\n', '|'), font=font, anchor='rb', fill=color)
            words.append(box.id)

        with test_data.write('words.txt') as f:
            f.write(', '.join(words))

        used.save(test_data.path("image.png"))

    @pytest.mark.parametrize('image_data', [
        pytest.param('test.png', id='test'),
        pytest.param('webcam.png', id='webcam'),
        pytest.param('Ortseingangsschild_Ilmenau.jpg', id='ortsschild')
    ], indirect=True)
    @pytest.mark.parametrize('module_args', [
        pytest.param({}, id="default"),
        pytest.param({'filter_empty_boxes': True}, id="empty"),
        pytest.param({
            'ocr_confidence': 0,
            'result_fmt': "{text} ({conf})",
            'filter_empty_boxes': False
        }, id="confvals"),
        pytest.param({
            'ocr_confidence': 50,
            'result_fmt': "{text} ({conf})",
            'filter_empty_boxes': False
        }, id="confvals-50"),
        pytest.param({'ocr_confidence': 50, 'filter_empty_boxes': False}, id="50"),
        pytest.param({'ocr_confidence': 50, 'filter_empty_boxes': True}, id="empty-50"),
        pytest.param({'ocr_confidence': 60, 'filter_empty_boxes': True}, id="empty-60"),
        pytest.param({'ocr_confidence': 65, 'filter_empty_boxes': True}, id="empty-65"),
    ])
    @pytest.mark.parametrize('base_type', PURE_SPECS + MSER_SPECS + EAST_SPECS)
    async def test_processing_module(
            self,
            image_data,
            event_loop: asyncio.AbstractEventLoop,
            run_time,
            base_type,
            module_args,
            ocr_module,
            topic_data,
            caplog,
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
        caplog.clear()

        start_time = event_loop.time()
        while len(topic_data) <= 50 and event_loop.time() - start_time < run_time:
            await asyncio.sleep(1. / ocr_module.processing_mode.frequency.hertz)
