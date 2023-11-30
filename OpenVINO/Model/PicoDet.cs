using OpenCvSharp;
using OpenVINO;
using OpenVINO.Results;
using OpenVinoSharp;
using static OpenVINO.MainWindow;

namespace OpenVinoSharpPPTinyPose
{
    public class PicoDet : OnnxModel
    {
        public PicoDet(string model_path, string device_name = "AUTO") : base(model_path, device_name)
        {
            
        }

        public override Result predict(Mat image)
        {
            image = image.Clone();

            Logger("\n/////////////////////////////////");
            Logger("===== ↓[PicoDet]↓ =====\n");

            Tensor input = infer.get_input_tensor();
            Size input_size = new Size(input.get_shape()[2], input.get_shape()[3]);

            Logger("====== [INPUT] ======");
            Logger($"Type: {input.get_element_type().to_string()}");
            Logger($"Shape: {input.get_shape().to_string()}");
            Logger($"Size: {input.get_size()}");

            // 求取缩放大小
            double scale_x = (double)image.Width / input_size.Width;
            double scale_y = (double)image.Height / input_size.Height;
            Point2d scale_factor = new Point2d(scale_x, scale_y);

            input.set_data(ImageToData.ImageToDataWithNormal(image, input_size, input.get_size()));

            // 模型推理
            Logger("==== >> 推理开始 << ====");
            infer.infer();
            Logger("==== >> 推理完成 << ==== \n");

            Tensor output = infer.get_output_tensor(0);
            Tensor output_1 = infer.get_output_tensor(1);

            Logger("====== [OUTPUT] ======");
            Logger($"Type: {output.get_element_type().to_string()}");
            Logger($"Shape: {output.get_shape().to_string()}");
            Logger($"Size: {output.get_size()}");

            Logger("====== [OUTPUT-1] ======");
            Logger($"Type: {output_1.get_element_type().to_string()}");
            Logger($"Shape: {output_1.get_shape().to_string()}");
            Logger($"Size: {output_1.get_size()}");

            // 读取置信值结果
            float[] results_con = output_1.get_data<float>((int) output_1.get_size());
            // 读取预测框
            float[] result_box = output.get_data<float>((int)output.get_size());

            BoxResult box = new BoxResult(scale_factor, 0.5f, 0.5f);

            // 处理模型推理数据
            return box.process_result(results_con, result_box);
        }

        public override void draw(Result result, Mat image)
        {
            for (int i = 0; i < result.rects.Count; i++)
            {
                Cv2.Rectangle(image, result.rects[i], new Scalar(255, 0, 0), 3);
                Cv2.PutText(image, result.scores[i].ToString(), new Point(result.rects[i].X, result.rects[i].Y - 5), HersheyFonts.Italic, 0.5, new Scalar(0, 0, 255));
            }
        }
    }
}
