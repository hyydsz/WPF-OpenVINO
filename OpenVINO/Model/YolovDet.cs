using OpenCvSharp;
using OpenVINO.Results;
using OpenVinoSharp;
using System.Collections.Generic;
using System.IO;
using static OpenVINO.MainWindow;

namespace OpenVINO.Model
{
    public class YolovDet : OnnxModel
    {
        public List<string> class_name_path = new List<string>();

        public YolovDet(string class_name_path, string model_path, string device_name = "AUTO") : base(model_path, device_name)
        {
            foreach (var item in File.ReadLines(class_name_path))
            {
                this.class_name_path.Add(item);
            }
        }

        public override Result predict(Mat image)
        {
            image = image.Clone();

            Logger("\n/////////////////////////////////");
            Logger("===== ↓[YolovDet]↓ =====\n");

            Tensor input = infer.get_input_tensor();
            Size input_size = new Size(input.get_shape()[2], input.get_shape()[3]);

            Logger("====== [INPUT] ======");
            Logger($"Type: {input.get_element_type().to_string()}");
            Logger($"Shape: {input.get_shape().to_string()}");
            Logger($"Size: {input.get_size()}\n");

            // 求取缩放大小
            double scale_x = (double)image.Width / input_size.Width;
            double scale_y = (double)image.Height / input_size.Height;
            Point2d scale_factor = new Point2d(scale_x, scale_y);

            input.set_data(ImageToData.ImageToDataWithNormal(image, input_size, input.get_size()));

            // 模型推理
            Logger("==== >> 推理开始 << ====");
            infer.infer();
            Logger("==== >> 推理完成 << ==== \n");

            Tensor output = infer.get_output_tensor();

            Logger("====== [OUTPUT] ======");
            Logger($"Type: {output.get_element_type().to_string()}");
            Logger($"Shape: {output.get_shape().to_string()}");
            Logger($"Size: {output.get_size()}");

            // 读取识别内容
            float[] result_data = output.get_data<float>((int)output.get_size());

            DetectionResult result = new DetectionResult(scale_factor, 0.25f);
            // 处理模型推理数据
            return result.process_result(result_data);
        }

        public override void draw(Result result, Mat image)
        {
            // 将识别结果绘制到图片上
            for (int i = 0; i < result.length; i++)
            {
                Cv2.Rectangle(image, result.rects[i], new Scalar(0, 0, 255), 2);

                string text = class_name_path[result.cls[i]] + "-" + result.scores[i].ToString("0.00");

                Cv2.Rectangle(image, 
                    new Point(result.rects[i].TopLeft.X, result.rects[i].TopLeft.Y - 20),
                    new Point(result.rects[i].X + Cv2.GetTextSize(text, HersheyFonts.HersheySimplex, 0.6, 1, out int baseLine).Width, result.rects[i].TopLeft.Y), 
                    new Scalar(255, 255, 255), -1);

                Cv2.PutText(image, text, 
                    new Point(result.rects[i].X, result.rects[i].Y - 5), HersheyFonts.HersheySimplex, 0.6, 
                    new Scalar(0, 0, 0));

            }
        }
    }
}
