"""
Consider setting your locale to C, e.g. by exporting LC_CTYPE=C
since tesseract code sometimes segfaults if the locale is set incorrectly.

Seems to be related to https://github.com/sirfz/tesserocr/issues/165 which seems to be fixed in
tesseract but not in the leptonica library see https://github.com/DanBloomberg/leptonica/issues/591
"""

from __future__ import annotations

import functools
import os
import typing
from functools import partial

import numpy as np

import ubii.proto

try:
    import importlib.resources as importlib_resources
except ImportError:
    # Try backported to PY<=37 `importlib_resources`.
    import importlib_resources  # noqa

try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property  # noqa

import logging
from tesserocr import PyTessBaseAPI, RIL, OEM, PSM
import cv2.cv2 as cv2

import ubii.proto as ub
from ubii.framework.processing import ProcessingRoutine

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

try:
    from PIL import Image
except ImportError:
    import Image

__protobuf__ = ub.__protobuf__

log = logging.getLogger(__name__)
perf_log = logging.getLogger(__name__ + '.performance')

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

PixelBox = typing.Tuple[int, int, int, int]


class BaseModule(ProcessingRoutine):
    """
    All OCR Modules inherit from this processing module.
    It supplies the basic functionality of loading images and
    transforming them, and defines the protobuf specs
    """

    def __init__(
            self,
            context,
            mapping: typing.Dict[str, typing.Any] = None,
            eval_strings: bool = False,
            api_variables: typing.Dict[str, typing.Any] = None,
            api_args: typing.Dict[str, typing.Any] = None,
            filter_empty_boxes: bool = False,
            ocr_confidence: int = 70,
            result_fmt: str | typing.Callable = '{text}',
            **kwargs
    ):
        """
        Create a processing module

        Args:
            context: Client context, contains broker constants definitions
            mapping: passed to protobuf initialization
            eval_strings: evaluate protobuf method definitions
            api_args: arguments passed to TesserOCR API initialization
            api_variables: mapping of variable names and values set in th api after initialization
            filter_empty_boxes: if the module should return bounding boxes where no text was detected
            ocr_confidence: cutoff for ocr detection, characters with lower confidence will be discarded
            result_fmt: This format string or callable defines how the 'id' field of the result will be formatted.
                Set it to a custom callable with kwargs to see which values will be passed to the callable
                (normally ocr result, confidence etc.)
            **kwargs: passed to protobuf initialization
        """
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
                'hertz': 60
            }
        }

        self._image: ubii.proto.Image2D | None = None
        self._cv2_image = None
        self._api: PyTessBaseAPI | None = None
        self._api_args = api_args
        self._api_vars = api_variables
        self._filter_empty_boxes = filter_empty_boxes
        self._log_performance_in_next_frame = False
        self._performance_values = []
        self._result_fmt = result_fmt
        self._ocr_confidence = ocr_confidence

    @property
    def filter_empty_boxes(self):
        return self._filter_empty_boxes

    @property
    def ocr_confidence(self):
        return self._ocr_confidence

    def on_init(self, context) -> None:
        """
        Create TesserOCR API instance
        """
        super().on_init(context)
        self._image = None
        self._api = PyTessBaseAPI(**(self._api_args or {}))

        for name, val in (self._api_vars or {}).items():
            self._api.SetVariable(name, val)
        log.info(f"Starting {self}")

    def read_image(self, image: ub.Image2D, conversion=bgr_conversions.get):
        """
        Read image data from protobuf message and set the image data used by
        the API instance. The image used by the API instance is always a grayscale
        version of the input image, as the Tesseract OCR performs better on grayscale images.

        Args:
            image: protobuf image
            conversion: method to get conversions from the :attr:`ubii.proto.Image2D.data_format`
                of the passed image to some format which is recognized by :func:`cv2.cvtColor`.
                You can pass None to not convert the image.

        Returns:
            result of the conversion
        """
        raw = np.frombuffer(
            bytearray(image.data), np.uint8
        ).reshape(
            (image.height, image.width, num_channels.get(image.data_format))
        )

        conversion = conversion(image.data_format)

        # tesserocr needs pillow image
        gray = cv2.cvtColor(raw, grayscale_conversions.get(image.data_format))
        self.api.SetImage(Image.fromarray(gray, mode='L'))

        self._cv2_image = cv2.cvtColor(raw, conversion) if conversion else raw
        return self._cv2_image

    def log_performance(self, context):
        """
        Just save some info of the module performance. Get's logged when module is destroyed since
        logging during processing would decrease performance significantly.
        """
        if self._log_performance_in_next_frame:
            self._performance_values.append(context.scheduler.exec_delta_times[-1])
            self._log_performance_in_next_frame = False

    @property
    def api(self) -> PyTessBaseAPI | None:
        """
        Reference to the loaded TesserOCR API instance
        """
        return self._api

    @property
    def image(self) -> ub.Image2D | None:
        """
        Reference to protobuf image if an image was received as input, else None
        """
        return self._image

    def on_created(self, context):
        super().on_created(context)

    def on_processing(self, context):
        """
        Load image from input, also save performance info
        """
        super().on_processing(context)
        self.log_performance(context)

        if context.inputs.image:
            self._image = context.inputs.image.image2D
            self._log_performance_in_next_frame = True

    def on_destroyed(self, context):
        """
        Unload API and log performance
        """
        import time
        time.sleep(1)

        del self._cv2_image
        del self._image
        self.api.__exit__(None, None, None)
        self._api = None

        log.info(f"Collected {len(self._performance_values)} execution times with input")
        perf_log.info(','.join(map('{:.4g}'.format, self._performance_values)))

        return super().on_destroyed(context)

    def ocr_in_box(self, box: PixelBox, min_confidence=70) -> str:
        """
        Perform OCR task using the tesseract api in a box inside the loaded image

        Args:
            box: region of interest
            min_confidence: cutoff to discard the OCR result

        Returns:
            recognized text in box
        """
        self.api.SetRectangle(*box)
        values = {'text': self.api.GetUTF8Text().strip(), 'conf': self.api.MeanTextConf()}
        if values['conf'] > min_confidence and values['text']:
            return self._result_fmt.format(**values) if isinstance(self._result_fmt, str) else self._result_fmt(**values)

    @staticmethod
    def object_2d_from_box(box: PixelBox) -> ub.Object2D:
        """
        Create protobuf message from integer tuple
        Args:
            box: box definition as (x, y, width, height)

        Returns:
            Protobuf message representing the box
        """
        x, y, w, h = box
        return ub.Object2D(
            pose={
                'position': {'x': x, 'y': y}
            },
            size={
                'x': w, 'y': h
            }
        )

    @staticmethod
    def to_image_space(dimensions: typing.Tuple[int, int], box: PixelBox) -> typing.Tuple[float, float, float, float]:
        """
        Convert absolute pixel coordinates to image space (i.e. coordinate in range [0, 1])
        Args:
            dimensions: image dimensions
            box: box coordinates

        Returns:
            box in image coordinates
        """
        width, height = dimensions
        x, y, w, h = box
        return x / width, y / height, w / width, h / height

    @staticmethod
    def box2rec(box: PixelBox) -> PixelBox:
        """
        Convert box in x, y, width, height format to coordinates of bottom left and top right corner points
        Args:
            box: tuple of x, y, width, height

        Returns:
            tuple of x1, y1, x2, y2
        """
        x, y, w, h = box
        return x, y, x + w, y + h

    @staticmethod
    def rec2box(rec: PixelBox) -> PixelBox:
        """
        Convert box in coordinates of bottom left and top right corner point to x, y, width, height
        Args:
            rec: tuple of x1, y1, x2, y2

        Returns:
            tuple of x, y, width, height
        """
        x1, y1, x2, y2 = rec
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def padded(box: PixelBox, padding: int = 0, dimensions: typing.Tuple[int, int] | None = None):
        """
        Expand box by padding in all directions
        Args:
            box: tuple of x, y, width, height
            padding: amount of pixels that should be padded around box.
            dimensions: used for clipping

        Returns:
            box: tuple of x, y, width, height for expanded box
        """
        x, y, w, h, = box
        padded = np.clip((x - padding, y - padding, w + padding * 2, h + padding * 2), a_min=0, a_max=None)

        x_max, y_max = dimensions
        max_vals = [x_max, y_max, x_max - padded[0], y_max - padded[1]]
        return np.clip(padded, a_min=None, a_max=max_vals)

    def __repr__(self):
        return f"{self.__class__}<processing_mode: {type(self.processing_mode).to_dict(self.processing_mode)}>"


class TesseractOCR_PURE(BaseModule):
    """
    This module uses pure Tesseract OCR functionality without preprocessing
    to extract text bounding boxes and perform OCR in them
    """

    def __init__(self, context, mapping=None, eval_strings=False, api_args=None, **kwargs):
        super().__init__(
            context, mapping, eval_strings, api_args=api_args or {'oem': OEM.DEFAULT, 'psm': PSM.AUTO_OSD},
            **kwargs
        )
        self.name = 'tesseract-ocr-pure'

    def on_processing(self, context):
        super().on_processing(context)
        predictions = ub.TopicDataRecord()

        if self.image:
            self.read_image(self.image)
            scale = partial(self.to_image_space, (self.image.width, self.image.height))
            pad = partial(self.padded, dimensions=(self.image.width, self.image.height))
            boxes = self.api.GetComponentImages(RIL.TEXTLINE, True)

            for _, region, _, _ in boxes:
                box = region['x'], region['y'], region['w'], region['h']
                result = self.object_2d_from_box(scale(box))
                result.id = self.ocr_in_box(pad(box, padding=10), min_confidence=self.ocr_confidence) or None
                if not self.filter_empty_boxes or result.id:
                    predictions.object2D_list.elements.append(result)

        if predictions.object2D_list.elements:
            context.outputs.predictions = predictions


class TesseractOCR_MSER(BaseModule):
    """
    This module uses the MSER algorithm to perform preprocessing and extract _character_ bounding
    boxes and performs OCR using Tesseract for the characters in those boxes
    """

    COLOR_ONLY_MSER_OPTS = [
        "min_diversity",
        "max_evolution",
        "area_threshold",
        "min_margin",
        "edge_blur_size"
    ]
    """
    These options are only usable with colored input images, if any are present in the dictionary
    passed as `mser_args` during initialization, the module will not use a grayscale conversion for the
    MSER algorithm which will result in longer processing times
    """

    def __init__(
            self,
            context,
            mser_grayscale: bool = True,
            padding=2,
            mapping=None,
            eval_strings=False,
            api_args=None,
            mser_args=None,
            **kwargs
    ):
        """
        Create a MSER preprocessing module

        Args:
            mser_grayscale: if True, run MSER algorithm on grayscale version of input image
            padding: the value used to pad all extracted regions before OCR is performed
            mser_args: passed to :func:`cv2.MSER_create`, default `{'max_variation': 0.25}`
        """

        super().__init__(
            context,
            mapping,
            eval_strings,
            api_args=api_args or {'oem': OEM.DEFAULT, 'psm': PSM.SINGLE_CHAR},
            **kwargs)
        # self.processing_mode.frequency = {'hertz': 20}
        self._mser = cv2.MSER_create(**(mser_args or {'max_variation': 0.25}))
        self._padding = padding
        self._mser_grayscale = mser_grayscale or any(opt in mser_args for opt in self.COLOR_ONLY_MSER_OPTS)
        self.name = 'tesseract-ocr-mser'

    def on_init(self, context) -> None:
        super().on_init(context)

    @classmethod
    def contained(cls, boxes: typing.Iterable[PixelBox]) -> typing.Iterable[bool]:
        """
        Compute if box is completely contained inside a box in passed boxes

        Args:
            box: box that should be queried
            boxes: List of boxes in x, y, width, height format

        Returns:
            items in list are True if box is contained inside another box, else False
        """

        def contains(rec1, rec2):
            """
            Return true if rec2 is completely inside rec1
            """
            r1, b1, l1, t1 = rec1
            r2, b2, l2, t2 = rec2
            return r1 <= r2 and l1 >= l2 and t1 >= t2 and b1 <= b2

        # This could be vectorized like in the EAST module
        rects = [cls.box2rec(box) for box in boxes]
        return [any(contains(r1, r2) for r1 in rects if r1 != r2) for r2 in rects]

    def on_processing(self, context):
        super().on_processing(context)
        if self.image:
            scale = partial(self.to_image_space, (self.image.width, self.image.height))
            pad = partial(self.padded, dimensions=(self.image.width, self.image.height))
            mser_input = self.read_image(
                self.image,
                **({'conversion': grayscale_conversions.get} if self._mser_grayscale else {})
            )
            _, bounding_boxes = self._mser.detectRegions(mser_input)
            bounding_boxes = np.unique(bounding_boxes, axis=0)
            contained = np.array(self.contained(bounding_boxes))
            predictions = ub.TopicDataRecord()
            for box in bounding_boxes[~contained, :]:
                box = pad(box, self._padding)
                result = self.object_2d_from_box(scale(box))
                result.id = self.ocr_in_box(box, min_confidence=self.ocr_confidence) or None
                if not self.filter_empty_boxes or result.id:
                    predictions.object2D_list.elements.append(result)

            if predictions.object2D_list.elements:
                context.outputs.predictions = predictions

    def on_destroyed(self, context):
        del self._mser
        super().on_processing(context)


class TesseractOCR_EAST(BaseModule):
    """
    This module uses the EAST algorithm to preprocess
    the image and extract text bounding boxes, then uses Tesseract for
    OCR inside the boxes
    """

    def __init__(
            self,
            context,
            mapping=None,
            eval_strings=False,
            api_args=None,
            nms_threshold: float = 0.5,
            min_detection_confidence: float = 0.7,
            merge_bounding_boxes: bool = False,
            **kwargs
    ):
        """

        Args:
            context: Client context, contains broker constants definitions
            mapping: passed to protobuf initialization
            eval_strings: evaluate protobuf method definitions
            api_args: arguments passed to TesserOCR API initialization
            nms_threshold: Threshold for non-maximum suppression of detected bounding boxes
            min_detection_confidence: Minimum confidence for EAST detection
            merge_bounding_boxes: determines if non-suppressed boxes get merged
            **kwargs: passed to :class:`.BaseModule` initialization
        """
        super().__init__(context,
                         mapping,
                         eval_strings,
                         api_args=api_args or {'oem': OEM.DEFAULT, 'psm': PSM.AUTO},
                         **kwargs)
        # self.processing_mode.frequency = {'hertz': 10}
        self._output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        from . import data
        with importlib_resources.path(data, "frozen_east_text_detection.pb") as east_model:
            self._detector = cv2.dnn.readNet(os.fspath(east_model))

        self._detector_input_shape = (320, 320)
        self.name = 'tesseract-ocr-east'
        self._nms_threshold = nms_threshold
        self._min_detection_confidence = min_detection_confidence
        self._merge_bounding_boxes = merge_bounding_boxes

    @property
    def nms_threshold(self):
        """
        Threshold for EAST detection
        """
        return self._nms_threshold

    @property
    def min_detection_confidence(self):
        """
        Confidence cutoff for detection
        """
        return self._min_detection_confidence

    def _merge_boxes(self, boxes):
        # this works because all values are greater 0!
        xmin = boxes[:, 0].astype(int)
        xmax = (boxes[:, 0] + boxes[:, 2]).astype(int)
        ymin = boxes[:, 1].astype(int)
        ymax = (boxes[:, 1] + boxes[:, 3]).astype(int)
        x_min_mat = np.einsum('i,j->ji', xmin, xmin)
        x_max_mat = np.einsum('i,j->ji', xmax, xmax)
        y_min_mat = np.einsum('i,j->ji', ymin, ymin)
        y_max_mat = np.einsum('i,j->ji', ymax, ymax)

        left_in_range = (x_min_mat >= (xmin * xmin)) & (x_min_mat <= (xmin * xmax))
        right_in_range = (x_max_mat <= (xmax * xmax)) & (x_max_mat >= (xmax * xmin))
        bot_in_range = (y_min_mat >= (ymin * ymin)) & (y_min_mat <= (ymin * ymax))
        top_in_range = (y_max_mat <= (ymax * ymax)) & (y_max_mat >= (ymax * ymin))

        contained = (
                (left_in_range & bot_in_range)
                | (left_in_range & top_in_range)
                | (right_in_range & bot_in_range)
                | (right_in_range & top_in_range)
        )

        sym_contained = contained | contained.T

        def boxvals(contained, index):
            mask = contained[index]
            x = np.min(xmin[mask])
            w = np.max(xmax[mask]) - x
            y = np.min(ymin[mask])
            h = np.max(ymax[mask]) - y
            return np.asarray([x, y, w, h]).T

        make_box = functools.partial(boxvals, sym_contained)

        unique = {tuple(row) for row in map(make_box, range(len(boxes)))}
        return np.vstack(tuple(unique))

    def on_processing(self, context):
        super().on_processing(context)
        if self.image:
            bgr = self.read_image(self.image)
            bounding_boxes = self.predict(
                bgr,
                input_shape=self._detector_input_shape,
                conf=self.min_detection_confidence,
                nms_threshold=self.nms_threshold
            )

            scale = partial(self.to_image_space, (self.image.width, self.image.height))
            pad = partial(self.padded, dimensions=(self.image.width, self.image.height))
            predictions = ub.TopicDataRecord()
            for num, box in enumerate(b.astype(int) for b in bounding_boxes):
                box = pad(box, padding=10)
                result = self.object_2d_from_box(scale(box))
                result.id = self.ocr_in_box(box, min_confidence=self.ocr_confidence) or None
                if not self.filter_empty_boxes or result.id:
                    predictions.object2D_list.elements.append(result)

            if predictions.object2D_list.elements:
                context.outputs.predictions = predictions

    def predict(self, image, input_shape=(320, 320), conf=0.7, nms_threshold=0.5):
        orig_h, orig_w = image.shape[:2]
        new_h, new_w = input_shape

        blob = cv2.dnn.blobFromImage(image, 1.0, input_shape, (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self._detector.setInput(blob)
        scores, geometry = self._detector.forward(self._output_layers)
        [boxes, confidences] = self.decode(scores, geometry, min_confidence=conf)
        if boxes.size == 0:
            return boxes

        def rescale(box):
            box[[0, 2]] *= orig_w / new_w
            box[[1, 3]] *= orig_h / new_h
            return box

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms_threshold)
        scaled = rescale(boxes[indices].T).T

        return self._merge_boxes(scaled) if self._merge_bounding_boxes else scaled

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

    def on_destroyed(self, context):
        del self._detector
        super().on_destroyed(context)
