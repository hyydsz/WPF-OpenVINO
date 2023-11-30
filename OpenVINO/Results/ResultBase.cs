using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenVINO.Results
{
    public abstract class ResultBase
    {
        // 图片信息  缩放比例h, 缩放比例height, width
        public Point2d scales;
        // 置信度阈值
        public float score_threshold;
        // 非极大值抑制阈值
        public float nms_threshold;

        public ResultBase(Point2d scales, float score_threshold = 0.4f, float nms_threshold = 0.5f)
        {
            this.scales = scales;
            this.score_threshold = score_threshold;
            this.nms_threshold = nms_threshold;
        }
    }
}
