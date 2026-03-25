using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Predictor
{
    public static class ListHelpers
    {
        public static float[,] ToArray(List<float[]> list) 
        {
            // Source - https://stackoverflow.com/a/3546181
            // Posted by Dirk Vollmar, modified by community. See post 'Timeline' for change history
            // Retrieved 2026-03-25, License - CC BY-SA 2.5

            float[,] arr = new float[list.Count, list[0].Length];
            for (int i = 0; i < list.Count; i++)
            {
                for (int j = 0; j < list[0].Length; j++)
                {
                    arr[i, j] = list[i][j];
                }
            }

            return arr;
        }

        public static float[,] ToArray(LinkedList<float[]> list) => ToArray(list.ToList());

        public static (float[] mean, float[] std) ComputeNorm(List<float[]> data)
        {
            const float eps = 1e-6f;

            int dim = data[0].Length;
            float[] mean = new float[dim];
            float[] std = new float[dim];

            for (int i = 0; i < dim; i++)
                mean[i] = data.Average(row => row[i]);

            for (int i = 0; i < dim; i++)
            {
                std[i] = (float)Math.Sqrt(data.Average(row => Math.Pow(row[i] - mean[i], 2)));

                if (std[i] < eps) std[i] = 1f;
            }

            return (mean, std);
        }

        public static (float[] mean, float[] std) ComputeNorm(LinkedList<float[]> data) => ComputeNorm(data.ToList());
    }
}
