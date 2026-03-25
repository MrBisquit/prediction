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
        public static (Tensor,Tensor) BuildTensors(string json)
        {
            var jd = JsonConvert.DeserializeObject<ReadingContainer>(File.ReadAllText(json))!;

            List<float[]> GivenFormattedData = new();
            LinkedList<float[]> PredictedFormattedData = new();

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
                }
                else hourskip++;

                GivenFormattedData.Add([
                    item.temp,
                    item.humidity,
                    item.pressure,
                    item.wind_speed * 10,
                    item.light,
                    (float)Math.Sin(item.wind_direction * Math.PI / 180),
                    (float)Math.Cos(item.wind_direction * Math.PI / 180),
                    item.Month,
                    hour
                ]);

                PredictedFormattedData.AddLast([
                    item.temp,
                    item.humidity,
                    item.pressure,
                    item.rain * 100,
                ]);
            }

            skipmedian /= hoursreached;

            // Rotate predicted data by two hours

            for (int i = 0; i < skipmedian * 2; i++)
            {
                // Source - https://stackoverflow.com/a/9948241
                // Posted by Jon Skeet, modified by community. See post 'Timeline' for change history
                // Retrieved 2026-03-25, License - CC BY-SA 4.0

                var first = PredictedFormattedData.First;
                PredictedFormattedData.RemoveFirst();
                PredictedFormattedData.AddLast(first!);
            }

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
