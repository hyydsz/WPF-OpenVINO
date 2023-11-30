using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Collections.Generic;
using System.Diagnostics;

namespace OpenVINO.Results
{
    public class DetectionResult : ResultBase
    {
        public DetectionResult(Point2d scales, float score_threshold = 0.4F, float nms_threshold = 0.5F) : base(scales, score_threshold, nms_threshold)
        {

        }

        public Result process_result(float[] result)
        {
            Mat result_data = new Mat(84, 8400, MatType.CV_32F, result);
            result_data = result_data.T();

            // 存放结果list
            List<Rect> position_boxes = new List<Rect>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();

            // 预处理输出结果
            for (int i = 0; i < result_data.Rows; i++)
            {
                Mat classes_scores = result_data.Row(i).ColRange(4, 84);
                Point max_classId_point, min_classId_point;

                // 获取一组数据中最大值及其位置
                Cv2.MinMaxLoc(classes_scores, out double min_score, out double max_score, out min_classId_point, out max_classId_point);

                // 置信度 0～1之间
                // 获取识别框信息
                if (max_score > 0.25)
                {
                    float cx = result_data.At<float>(i, 0);
                    float cy = result_data.At<float>(i, 1);
                    float ow = result_data.At<float>(i, 2);
                    float oh = result_data.At<float>(i, 3);

                    int x = (int)((cx - 0.5 * ow) * scales.X);
                    int y = (int)((cy - 0.5 * oh) * scales.Y);
                    int width = (int)(ow * scales.X);
                    int height = (int)(oh * scales.Y);

                    Rect box = new Rect();
                    box.X = x;
                    box.Y = y;
                    box.Width = width;
                    box.Height = height;

                    position_boxes.Add(box);
                    class_ids.Add(max_classId_point.X);
                    confidences.Add((float) max_score);
                }
            }

            // NMS非极大值抑制
            CvDnn.NMSBoxes(position_boxes, confidences, score_threshold, nms_threshold, out int[] indexes);
            Result re_result = new Result(this);

            // 将识别结果绘制到图片上
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                re_result.add(confidences[index], position_boxes[index], class_ids[index]);
            }

            return re_result;
        }
    }
}
