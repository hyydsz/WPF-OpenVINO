using OpenCvSharp;
using System.Collections.Generic;
using System;
using OpenVINO.Results;
using OpenCvSharp.Dnn;

public class PoseResult : ResultBase
{
    /// <summary>
    /// 结果处理类构造
    /// </summary>
    /// <param name="scales">缩放比例</param>
    /// <param name="score_threshold">分数阈值</param>
    /// <param name="nms_threshold">非极大值抑制阈值</param>
    public PoseResult(Point2d scales, float score_threshold = 0.25f, float nms_threshold = 0.5f) : base(scales, score_threshold, nms_threshold)
    {

    }

    /// <summary>
    /// 结果处理
    /// </summary>
    /// <param name="result">模型预测输出</param>
    /// <returns>模型识别结果</returns>
    public Result process_result(float[] result)
    {
        // Row -> Height
        // Col -> Width

        // Row Col
        Mat result_data = new Mat(56, 8400, MatType.CV_32F, result);
        result_data = result_data.T();

        // 存放结果list
        List<Rect> position_boxes = new List<Rect>();
        List<float> confidences = new List<float>();
        List<PoseData> pose_datas = new List<PoseData>();

        // 预处理输出结果
        for (int i = 0; i < result_data.Rows; i++)
        {
            // 获取识别框和关键点信息
            if (result_data.At<float>(i, 4) > score_threshold)
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

                Mat pose_mat = result_data.Row(i).ColRange(5, 56);
                pose_mat.GetArray(out float[] pose_data);

                PoseData pose = new PoseData(pose_data, scales);

                position_boxes.Add(box);
       
                confidences.Add(result_data.At<float>(i, 4));
                pose_datas.Add(pose);
            }
        }

        // NMS非极大值抑制
        CvDnn.NMSBoxes(position_boxes, confidences, score_threshold, nms_threshold, out int[] indexes);

        Result re_result = new Result(this);

        // 将识别结果绘制到图片上
        for (int i = 0; i < indexes.Length; i++)
        {
            int index = indexes[i];
            re_result.add(confidences[index], position_boxes[index], pose_datas[i]);
        }

        return re_result;
    }

    public class PoseData
    {
        public float[] score = new float[17];
        public List<Point> point = new List<Point>();

        public PoseData(float[] data, Point2d scales)
        {
            for (int i = 0; i < 17; i++)
            {
                Point p = new Point((int)(data[3 * i] * scales.X), (int)(data[3 * i + 1] * scales.Y));

                this.point.Add(p);
                this.score[i] = data[3 * i + 2];
            }
        }
    }
}