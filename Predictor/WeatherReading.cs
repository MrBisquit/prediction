using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using static TorchSharp.torch;

namespace Predictor
{
    public class ReadingContainer
    {
        public static (Tensor,Tensor) BuildTensors(string json,int hourspredict)
        {
            ReadingContainer jd;

            if ((File.GetAttributes(json) & FileAttributes.Directory) == FileAttributes.Directory)
            {
                List<WeatherReading> readings = new();

                foreach (var item in Directory.GetFiles(json,"*.json",SearchOption.AllDirectories))
                {
                    var tjd = JsonConvert.DeserializeObject<ReadingContainer>(File.ReadAllText(item))!;

                    readings.AddRange(tjd.points);
                }

                jd = new()
                {
                    points = readings.ToArray()
                };
            }
            else jd = JsonConvert.DeserializeObject<ReadingContainer>(File.ReadAllText(json))!;

            List<float[]> GivenFormattedData = new();
            List<float[]> PredictedFormattedData = new();

            var hourskip = 0;
            var skipmedian = 0;
            var hoursreached = 0;

            var pasthour = 0;

            // build given data + calculate readings per hour median

            foreach (var item in jd.points)
            {
                var hour = DateTime.Parse(item.date).Hour;

                if (hour != pasthour)
                {
                    skipmedian += hourskip;
                    hourskip = 0;
                    hoursreached++;

                    pasthour = hour;
                }
                else hourskip++;

                GivenFormattedData.Add([
                    item.temp,
                    item.humidity,
                    item.pressure,
                    item.wind_speed,
                    item.light,
                    (float)Math.Sin(item.wind_direction * Math.PI / 180),
                    (float)Math.Cos(item.wind_direction * Math.PI / 180),
                    item.Month,
                    hour
                ]);

                PredictedFormattedData.Add([
                    item.temp,
                    item.humidity,
                    item.pressure,
                    item.rain > 0 ? 1 : 0,
                ]);
            }

            skipmedian /= hoursreached;

            // Rotate predicted data by hourspredict hours

            LinkedList<float[]> shuffler = new(PredictedFormattedData);

            for (int i = 0; i < skipmedian * hourspredict; i++)
            {
                // Source - https://stackoverflow.com/a/9948241
                // Posted by Jon Skeet, modified by community. See post 'Timeline' for change history
                // Retrieved 2026-03-25, License - CC BY-SA 4.0

                var first = shuffler.First;
                shuffler.RemoveFirst();
                shuffler.AddLast(first!);
            }

            PredictedFormattedData = shuffler.ToList();

            var (inMean, inStd) = ListHelpers.ComputeNorm(GivenFormattedData);
            var (outMean, outStd) = ListHelpers.ComputeNorm(PredictedFormattedData);

            for (int r = 0; r < GivenFormattedData.Count; r++)
                for (int c = 0; c < 9; c++)
                    GivenFormattedData[r][c] = (GivenFormattedData[r][c] - inMean[c]) / inStd[c];

            for (int r = 0; r < PredictedFormattedData.Count; r++)
                for (int c = 0; c < 4; c++)
                    PredictedFormattedData[r][c] = (PredictedFormattedData[r][c] - outMean[c]) / outStd[c];

            File.WriteAllText("data_normals.json",
                JsonConvert.SerializeObject(new NormalizedJson() {
                    inMean = inMean, inStd = inStd,
                    outMean = outMean, outStd = outStd,
                }));

            return (
                tensor(ListHelpers.ToArray(GivenFormattedData)).reshape(GivenFormattedData.Count,9),
                tensor(ListHelpers.ToArray(PredictedFormattedData)).reshape(GivenFormattedData.Count,4));
        }

        public WeatherReading[] points { get; set; } = [];
    }

    public class WeatherReading
    {
        public string date { get; set; } = "2026-02-14T01:00:05.000Z"; // temp
        public float temp { get; set; } = 0; // predicted, given
        public float humidity { get; set; } = 0; // predited, given
        public float pressure { get; set; } = 0; // predited, given
        public float wind_speed { get; set; } = 0; // given (processed, scaled)
        public float light { get; set; } = 0; // given
        public float rain { get; set; } = 0; // predicted (processed, scaled)
        public float wind_direction { get; set; } = 0; // given (processed, sine + cos)
        public float Month { get; set; } = 0; // given

        public static WeatherReading TestReading = new() 
        {
            temp = 2.12f,
            humidity = 68.39f,
            pressure = 976.1f,
            light = 0,
            wind_speed = 0.6667693f * 10,
            rain = 0,
            wind_direction = 315,
            Month = 2
        };
    }
}
