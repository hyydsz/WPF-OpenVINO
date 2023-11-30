using OpenCvSharp;
using OpenVINO;
using OpenVINO.Results;
using OpenVinoSharp;
using System;
using static OpenVINO.MainWindow;

namespace OpenVinoSharpPPTinyPose
{
    public class PPTinyPose : OnnxModel
    {
        public PPTinyPose(string model_path, string device_name = "AUTO") : base(model_path, device_name)
        {

        }

        public override Result predict(Mat image)
        {
            image = image.Clone();

            Logger("\n/////////////////////////////////");
            Logger("===== ↓[TinyPose]↓ =====\n");

            Tensor input = infer.get_input_tensor();
            Size input_size = new Size(input.get_shape()[2], input.get_shape()[3]);

            // 求取缩放大小
            double scale_x = (double)image.Width / input_size.Width;
            double scale_y = (double)image.Height / input_size.Height;
            Point2d scale_factor = new Point2d(scale_x, scale_y);

            Logger("====== [INPUT] ======");
            Logger($"Type: {input.get_element_type().to_string()}");
            Logger($"Shape: {input.get_shape().to_string()}");
            Logger($"Size: {input.get_size()}\n");

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

            float[] result = output.get_data<float>((int) output.get_size());

            PoseResult poseResult = new PoseResult(scale_factor, 0.25f, 0.5f);  
            Result points = poseResult.process_result(result);

            return points;
        }

        public override void draw(Result result, Mat image)
        {
            

            foreach (var pose in result.poses)
            {
                // 连接点关系
                int[,] edgs = new int[17, 2] { { 0, 1 }, { 0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}, {6, 8},
                 {7, 9}, {8, 10}, {5, 11}, {6, 12}, {11, 13}, {12, 14},{13, 15 }, {14, 16 }, {11, 12 } };

                // 颜色库
                Scalar[] colors = new Scalar[18] { new Scalar(255, 0, 0), new Scalar(255, 85, 0), new Scalar(255, 170, 0),
                new Scalar(255, 255, 0), new Scalar(170, 255, 0), new Scalar(85, 255, 0), new Scalar(0, 255, 0),
                new Scalar(0, 255, 85), new Scalar(0, 255, 170), new Scalar(0, 255, 255), new Scalar(0, 170, 255),
                new Scalar(0, 85, 255), new Scalar(0, 0, 255), new Scalar(85, 0, 255), new Scalar(170, 0, 255),
                new Scalar(255, 0, 255), new Scalar(255, 0, 170), new Scalar(255, 0, 85) };

                for (int p = 0; p < 17; p++)
                {
                    if (pose.score[p] < result.score_threshold)
                    {
                        continue;
                    }

                    Cv2.Circle(image, pose.point[p], 2, colors[p], -1);
                }

                // 绘制
                for (int p = 0; p < 17; p++)
                {
                    if (pose.score[edgs[p, 0]] < result.score_threshold || pose.score[edgs[p, 1]] < result.score_threshold)
                    {
                        continue;
                    }

                    float[] point_x = new float[] { pose.point[edgs[p, 0]].X, pose.point[edgs[p, 1]].X };
                    float[] point_y = new float[] { pose.point[edgs[p, 0]].Y, pose.point[edgs[p, 1]].Y };

                    Point center_point = new Point((point_x[0] + point_x[1]) / 2, (point_y[0] + point_y[1]) / 2);

                    double length = Math.Sqrt(Math.Pow(point_x[0] - point_x[1], 2.0) + Math.Pow(point_y[0] - point_y[1], 2.0));
                    int stick_width = 2;

                    Size axis = new Size(length / 2, stick_width);

                    double angle = Math.Atan2(point_y[0] - point_y[1], point_x[0] - point_x[1]) * 180 / Math.PI;
                    Point[] polygon = Cv2.Ellipse2Poly(center_point, axis, (int)angle, 0, 360, 1);

                    Cv2.FillConvexPoly(image, polygon, colors[p]);
                }
            }
        }
    }
}
