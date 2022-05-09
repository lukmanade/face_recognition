import 'package:camera/camera.dart';
import 'package:face_recognition/faceModule/utils.dart';
import 'package:image/image.dart' as imglib;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:google_ml_kit/google_ml_kit.dart';

Future<tfl.Interpreter?> loadModel() async {
  try {
    final gpuDelegateV2 = tfl.GpuDelegateV2(
        options: tfl.GpuDelegateOptionsV2(
            isPrecisionLossAllowed: false,
            inferencePreference: tfl.TfLiteGpuInferenceUsage.fastSingleAnswer,
            inferencePriority1: tfl.TfLiteGpuInferencePriority.minMemoryUsage,
            inferencePriority2: tfl.TfLiteGpuInferencePriority.minLatency,
            inferencePriority3: tfl.TfLiteGpuInferencePriority.auto,
            maxDelegatePartitions: 1));

    var interpreterOptions = tfl.InterpreterOptions()
      ..addDelegate(gpuDelegateV2);
    return await tfl.Interpreter.fromAsset('mobilefacenet.tflite',
        options: interpreterOptions);
  } on Exception {
    print('Failed to load model.');
    return null;
  }
}

imglib.Image convertCameraImage(CameraImage image, CameraLensDirection _dir) {
  int width = image.width;
  int height = image.height;
  var img = imglib.Image(width, height); // buat gambar buffer
  const int hexFF = 0xFF000000;
  final int uvyButtonStride = image.planes[1].bytesPerRow;
  final int? uvPixelStride = image.planes[1].bytesPerPixel;
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      final int uvIndex =
          uvPixelStride! * (x / 2).floor() + uvyButtonStride * (y / 2).floor();
      final int index = y * width + x;
      final yp = image.planes[0].bytes[index];
      final up = image.planes[1].bytes[uvIndex];
      final vp = image.planes[2].bytes[uvIndex];
      // Calculate pixel color
      int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
      int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
          .round()
          .clamp(0, 255);
      int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
      // color: 0x FF  FF  FF  FF
      //           A   B   G   R
      img.data[index] = hexFF | (b << 16) | (g << 8) | r;
    }
  }
  var img1 = (_dir == CameraLensDirection.front)
      ? imglib.copyRotate(img, -90)
      : imglib.copyRotate(img, 90);
  return img1;
}

HandleDetection getDetectionMethod() {
  //deteksi wajah atau bukan
  final faceDetector = GoogleMlKit.vision.faceDetector(
    const FaceDetectorOptions(mode: FaceDetectorMode.accurate),
  );
  return faceDetector.processImage;
}
