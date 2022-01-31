from __future__ import annotations

import os
from functools import wraps, partial

import numpy as np

try:
    import importlib.resources as importlib_resources
except ImportError:
    # Try backported to PY<=37 `importlib_resources`.
    import importlib_resources

try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property

import logging
from tesserocr import PyTessBaseAPI, RIL, OEM, PSM
import cv2.cv2 as cv2  # pycharm is stupid

import ubii.proto as ub
from ubii.framework.processing import ProcessingRoutine

try:
    from PIL import Image
except ImportError:
    import Image

__protobuf__ = ub.__protobuf__

log = logging.getLogger(__name__)

num_channels = {
    ub.Image2D.DataFormat.RGB8: 3,
    ub.Image2D.DataFormat.RGBA8: 4,
    ub.Image2D.DataFormat.GRAY8: 1,
}
grayscale_conversions = {
    ub.Image2D.DataFormat.RGB8: cv2.COLOR_RGB2GRAY,
    ub.Image2D.DataFormat.GRAY8: None,
    ub.Image2D.DataFormat.RGBA8: cv2.COLOR_RGBA2GRAY
}

bgr_conversions = {
    ub.Image2D.DataFormat.RGB8: cv2.COLOR_RGB2BGR,
    ub.Image2D.DataFormat.GRAY8: cv2.COLOR_GRAY2BGR,
    ub.Image2D.DataFormat.RGBA8: cv2.COLOR_RGBA2BGR,
}


class RunOnFrame:
    def __init__(self, num_frames, callback):
        self.processing_count = 0
        self.num_frames = num_frames
        self.callback = callback

    def __call__(self, processing_callback):
        @wraps(processing_callback)
        def wrapped(processing_routine, context):
            self.processing_count += 1
            if self.processing_count % self.num_frames == 0:
                self.callback(processing_routine, context)
                self.processing_count = 0

            processing_callback(processing_routine, context)

        return wrapped


class BaseModule(ProcessingRoutine):

    def __init__(self, context, mapping=None, eval_strings=False, api_args=None, **kwargs):
        super().__init__(mapping, eval_strings, **kwargs)
        constants = context.constants
        self.tags = ['ocr', 'text-detection', 'tesseract']
        self.description = 'Trying some OCR Stuff'

        self.inputs = [
            {
                'internal_name': 'image',
                'message_format': constants.MSG_TYPES.DATASTRUCTURE_IMAGE
            },
        ]

        self.outputs = [
            {
                'internal_name': 'predictions',
                'message_format': constants.MSG_TYPES.DATASTRUCTURE_OBJECT2D_LIST
            },
        ]

        self.processing_mode = {
            'frequency': {
                'hertz': 5
            }
        }

        self._image = None
        self._api = PyTessBaseAPI(**(api_args or {}))

    def read_image(self, image: ub.Image2D, conversion=bgr_conversions.get):
        data = np.frombuffer(
            bytearray(image.data), np.uint8
        ).reshape(
            (image.height, image.width, num_channels.get(image.data_format))
        )
        conversion = conversion(image.data_format)
        if conversion:
            data = cv2.cvtColor(data, conversion)

        gray = cv2.cvtColor(data, grayscale_conversions.get(image.data_format))
        # tesserocr needs pillow image
        self.api.SetImage(Image.fromarray(gray, mode='L'))

        return data

    @property
    def api(self):
        return self._api

    @property
    def image(self):
        return self._image

    def on_created(self, context):
        super().on_created(context)
        self._loop = context.loop

    def log_performace(self, context):
        log.info(f"Performance: {context.scheduler.performance_rating:.2%}")

    def ocr_in_box(self, box, min_confidence=70):
        self.api.SetRectangle(*box)
        ocr_result = self.api.GetUTF8Text().strip()
        conf = self.api.MeanTextConf()
        if conf > min_confidence and ocr_result:
            return ocr_result

    @staticmethod
    def object_2d_from_box(box):
        x, y, w, h = box
        return ub.Object2D(
            pose={
                'position': {'x': x, 'y': y}
            },
            size={
                'x': w, 'y': h
            }
        )

    def to_image_space(self, dimensions, box):
        width, height = dimensions
        x, y, w, h = box
        return x / width, y / height, w / width, h / height

    def on_processing(self, context):
        super().on_processing(context)
        self._image = context.inputs.image

    def on_destroyed(self, context):
        self.api.__exit__(None, None, None)
        self._executor_pool.shutdown(wait=True)
        return super().on_destroyed(context)

    @staticmethod
    def box2rec(box):
        x, y, w, h = box
        return x, y, x + w, y + h

    @staticmethod
    def rec2box(rec):
        x1, y1, x2, y2 = rec
        return x1, y1, x2 - x1, y2 - y1


class TesseractOCR_PURE(BaseModule):


    def __init__(self, context, mapping=None, eval_strings=False, api_args=None, **kwargs):
        super().__init__(context, mapping, eval_strings, api_args, **kwargs)
        self.name = 'tesseract-ocr-pure'

    def on_processing(self, context):
        super().on_processing(context)
        results = []
        if self.image:
            loaded = Image.frombuffer('RGB', (self.image.width, self.image.height), self.image.data)
            scale = partial(self.to_image_space, (self.image.width, self.image.height))
            self.api.SetImage(loaded)
            boxes = self.api.GetComponentImages(RIL.TEXTLINE, True)

            for i, (im, box, _, _) in enumerate(boxes):
                text = self.ocr_in_box((box['x'], box['y'], box['w'], box['h']))
                if text:
                    result = self.object_2d_from_box(scale((box['x'], box['y'], box['w'], box['h'])))
                    result.id = text
                    results.append(result)

        context.outputs.predictions = ub.Object2DList(elements=results)


class TesseractOCR_MSER(BaseModule):

    def __init__(self, context, mapping=None, eval_strings=False, **kwargs):
        super().__init__(
            context,
            mapping,
            eval_strings,
            api_args={'oem': OEM.DEFAULT, 'psm': PSM.SINGLE_CHAR},
            **kwargs)
        self.processing_mode.frequency = {'hertz': 10}
        self._mser = cv2.MSER_create(max_variation=0.25)
        self.name = 'tesseract-ocr-mser'

    @RunOnFrame(num_frames=30, callback=BaseModule.log_performace)
    def on_processing(self, context):
        super().on_processing(context)
        if self.image:
            scale = partial(self.to_image_space, (self.image.width, self.image.height))
            gray = self.read_image(self.image, conversion=grayscale_conversions.get)
            _, bounding_boxes = self.data.mser.detectRegions(gray)

            results = []
            for box in bounding_boxes:
                text = self.ocr_in_box(box)
                if text:
                    result = self.object_2d_from_box(scale(box))
                    result.id = text
                    results.append(result)

            if results:
                context.outputs.predictions = ub.Object2DList(elements=results)
                # log.info(f"Detected Text[s]:\n" + '\n'.join(map('{!r}'.format, results)))


class TesseractOCR_EAST(BaseModule):

    def __init__(self, context, mapping=None, eval_strings=False, api_args=None, **kwargs):
        super().__init__(context,
                         mapping,
                         eval_strings,
                         api_args=api_args or {'oem': OEM.DEFAULT, 'psm': PSM.AUTO},
                         **kwargs)
        self.processing_mode.frequency = {'hertz': 10}
        self._output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        from . import data
        with importlib_resources.path(data, "frozen_east_text_detection.pb") as east_model:
            self._detector = cv2.dnn.readNet(os.fspath(east_model))

        self._detector_input_shape = (320, 320)
        self.name = 'tesseract-ocr-east'

    @RunOnFrame(num_frames=30, callback=BaseModule.log_performace)
    def on_processing(self, context):
        super().on_processing(context)
        if self.image:
            bgr = self.read_image(self.image)
            bounding_boxes = self.predict(bgr, input_shape=self._detector_input_shape)
            scale = partial(self.to_image_space, (self.image.width, self.image.height))
            results = []
            for box in bounding_boxes:
                padding = 40
                x, y, w, h = box.astype(int)
                padded = x - padding // 2, y - padding // 2, w + padding, h + padding
                result = self.object_2d_from_box(scale(padded))
                result.id = self.ocr_in_box(padded)
                results.append(result)

            if results:
                context.outputs.predictions = ub.Object2DList(elements=results)
                # log.info(f"Detected Text[s]:\n" + '\n'.join(map('{!r}'.format, results)))

    def to_image_space(self, dimensions, box):
        x, y, w, h = super().to_image_space(dimensions, box)
        return x * 0.5, y, w, h  # opencv coordinate system

    def predict(self, image, input_shape=(320, 320), conf=0.7, nms_threshold=0.5):
        orig_h, orig_w = image.shape[:2]
        new_h, new_w = input_shape

        blob = cv2.dnn.blobFromImage(image, 1.0, input_shape, (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self._detector.setInput(blob)
        scores, geometry = self._detector.forward(self._output_layers)
        [boxes, confidences] = self.decode(scores, geometry, conf)
        if boxes.size == 0:
            return boxes

        def rescale(box):
            converted = box.astype(float)
            converted[[0, 2]] *= orig_w / new_w
            converted[[1, 3]] *= orig_h / new_h
            return converted

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms_threshold)
        return np.apply_along_axis(rescale, 1, boxes[indices])

    @staticmethod
    def decode(prob_score, geo, min_confidence=0.6):
        shape = prob_score.shape[2:4]
        confidence = prob_score[0, 0, :]
        mask = confidence >= min_confidence

        if not np.any(mask):
            return np.array([]), np.array([])

        x0 = geo[0, 0, mask]
        x1 = geo[0, 1, mask]
        x2 = geo[0, 2, mask]
        x3 = geo[0, 3, mask]

        heights = x0 + x2
        widths = x1 + x3
        angles = geo[0, 4, mask]

        offsets = np.indices(shape) * 4

        end_points = np.array([offsets[1, mask] + np.cos(angles) * x1 + np.sin(angles) * x2,
                               offsets[0, mask] - np.sin(angles) * x1 + np.cos(angles) * x2])

        start_points = np.array([end_points[0] - widths, end_points[1] - heights])
        # return bounding boxes and associated confidence_val
        return np.vstack((start_points, widths, heights)).T, confidence[mask]
