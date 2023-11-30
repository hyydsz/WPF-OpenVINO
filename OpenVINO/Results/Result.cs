using OpenCvSharp;
using OpenVINO.Results;
using System.Collections.Generic;

public class Result
{
    private ResultBase result_base;

    public Result(ResultBase result) {
        result_base = result;
    }

    // 置信度阈值
    public float score_threshold
    {
        get { return result_base.score_threshold; }
    }

    // 非极大值抑制阈值
    public float nms_threshold
    {
        get { return result_base.nms_threshold; } 
    }

    // 获取结果长度
    public int length
    {
        get
        {
            return scores.Count;
        }
    }
    // 识别物体序号
    public List<int> cls = new List<int>();
    // 置信值
    public List<float> scores = new List<float>();
    // 预测框
    public List<Rect> rects = new List<Rect>();
    // 分割区域
    public List<Mat> masks = new List<Mat>();
    // 人体关键点
    public List<PoseResult.PoseData> poses = new List<PoseResult.PoseData>();


    public void add(float score, Rect rect)
    {
        scores.Add(score);
        rects.Add(rect);
    }

    public void add(float score, Rect rect, Mat mask)
    {
        scores.Add(score);
        rects.Add(rect);
        masks.Add(mask);
    }

    public void add(float score, Rect rect, int name)
    {
        scores.Add(score);
        rects.Add(rect);
        cls.Add(name);
    }

    public void add(float score, Rect rect, PoseResult.PoseData pose)
    {
        scores.Add(score);
        rects.Add(rect);
        poses.Add(pose);
    }

    public void add(float score, PoseResult.PoseData pose)
    {
        scores.Add(score);
        poses.Add(pose);
    }
}