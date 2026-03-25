using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Predictor
{
    public class NormalizedJson
    {
        public float[] inMean { get; set; } = [];
        public float[] outMean { get; set; } = [];
        public float[] inStd { get; set; } = []; 
        public float[] outStd { get; set; } = [];
    }
}
