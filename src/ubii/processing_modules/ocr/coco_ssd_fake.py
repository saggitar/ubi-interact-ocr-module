import logging

from tesserocr import PyTessBaseAPI, RIL, PSM, OEM
from ubii.interact.processing import ProcessingRoutine

import ubii.proto as ub

try:
    from PIL import Image
except ImportError:
    import Image

__protobuf__ = ub.__protobuf__

log = logging.getLogger(__name__)


class CocoSSDPM(ProcessingRoutine):
    def __init__(self, context, mapping=None, eval_strings=False, **kwargs):
        super().__init__(mapping, eval_strings, **kwargs)
        constants = context.constants
        self.name = 'coco-ssd-object-detection'
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
                'hertz': 10
            }
        }

        self._image = None
        self._api = PyTessBaseAPI()
        self._processing_count = 0

    @property
    def api(self):
        return self._api

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value

    def on_processing(self, context):
        image = context.inputs.image

        self._processing_count += 1
        if self._processing_count % 30 == 0:
            log.info(f"Performance: {context.scheduler.performance_rating:.2%}")
            self._processing_count = 0

        if image:
            loaded = Image.frombuffer('RGB', (image.width, image.height), image.data)
            self.api.SetImage(loaded)
            boxes = self.api.GetComponentImages(RIL.TEXTLINE, True)
            result = []

            for i, (im, box, _, _) in enumerate(boxes):
                self.api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                ocrResult = self.api.GetUTF8Text()
                conf = self.api.MeanTextConf()
                if conf > 75 and ocrResult:
                    result.append(
                        ub.Object2D(
                            id=ocrResult.strip(),
                            pose={
                                'position': {'x': box['x'] / image.width, 'y': box['y'] / image.height}
                            },
                            size={
                                'x': box['w'] / image.width, 'y': box['h'] / image.height
                            }
                        )
                    )

            if result:
                context.outputs.predictions = ub.Object2DList(elements=result)
                log.info(f"Detected Text[s]:\n" + '\n'.join(map('{!r}'.format, result)))

    def on_destroyed(self, context):
        self.api.__exit__(None, None, None)
        return super().on_destroyed(context)
