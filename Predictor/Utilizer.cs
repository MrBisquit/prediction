using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;

namespace Predictor
{
    public class Utilizer
    {
        public static void Predict(WeatherReading reading,int hour,string modelpath)
        {

            List<float[]> GivenFormattedData = [[
                reading.temp,
                reading.humidity,
                reading.pressure,
                reading.wind_speed,
                reading.light,
                (float)Math.Sin(reading.wind_direction * Math.PI / 180),
                (float) Math.Cos(reading.wind_direction * Math.PI / 180),
                reading.Month,
                hour
            ]];

            var y = tensor(ListHelpers.ToArray(GivenFormattedData)).reshape(1, 9);

            var model = new WeatherNet();
            model.load(modelpath);

            using var _ = no_grad();
            var result = model.forward(y);

            float[] values = result.data<float>().ToArray();

            foreach (var v in values)
                Console.WriteLine(Math.Round(v));
        }
    }
}
