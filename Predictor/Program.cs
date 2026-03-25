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
                if (args.Length != 3)
                {
                    Console.WriteLine("Invalid argument count for 'train'");
                    return;
                }

                Trainer.Train(args[1], args[2]);
                break;
            default:
                Console.WriteLine($"Invalid command: {args[0]}");
                break;
        }
    }
}