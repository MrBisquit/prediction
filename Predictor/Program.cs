using Predictor;

public static class Program
{
    public static void Main(string[] args)
    {
        if (args.Length == 0) 
        {
            Console.WriteLine("usage: Predictor [command] [parameters] <arguments>");
            return;
        }

        switch (args[0])
        {
            case "train":
                if (args.Length != 5)
                {
                    Console.WriteLine("Invalid argument count for 'train'");
                    return;
                }

                Trainer.Train(args[1], args[2],int.Parse(args[3]), int.Parse(args[4]));
                break;
            case "test":
                if (args.Length != 2)
                {
                    Console.WriteLine("Invalid argument count for 'test'");
                    return;
                }

                Utilizer.Predict(WeatherReading.TestReading, 0, args[1]);
                break;
            default:
                Console.WriteLine($"Invalid command: {args[0]}");
                break;
        }
    }
}