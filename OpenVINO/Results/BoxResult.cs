using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Collections.Generic;

namespace OpenVINO.Results
{
    public class BoxResult : ResultBase
    {
        public BoxResult(Point2d scales, float score_threshold = 0.4F, float nms_threshold = 0.5F) : base(scales, score_threshold, nms_threshold)
        {

        }

        /// <summary>
        /// 结果处理
        /// </summary>
        /// <param name="result">模型预测输出</param>
        /// <returns>模型识别结果</returns>
        public Result process_result(float[] results_con, float[] result_box)
        {
            // 处理预测结果
            List<float> confidences = new List<float>();
            List<Rect> boxes = new List<Rect>();

            Result re_result = new Result(this);

            for (int c = 0; c < result_box.Length / 4; c++)
            {
                // 重新构建
                Rect rect = new Rect(
                    (int)(result_box[4 * c] * scales.X),
                    (int)(result_box[4 * c + 1] * scales.Y),
                    (int)((result_box[4 * c + 2] - result_box[4 * c]) * scales.X),
                    (int)((result_box[4 * c + 3] - result_box[4 * c + 1]) * scales.Y)
                    );

                boxes.Add(rect);
                confidences.Add(results_con[c]);
            }

            // 非极大值抑制获取结果候选框
            CvDnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold, out int[] indexes);

            for (int i = 0; i < indexes.Length; i++)
            {
                re_result.add(confidences[indexes[i]], boxes[indexes[i]]);
            }

            return re_result;
        }
    }
}
