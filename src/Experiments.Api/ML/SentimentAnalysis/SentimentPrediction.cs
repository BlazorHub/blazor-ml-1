using Microsoft.ML.Data;

namespace Experiments.Api.ML.SentimentAnalysis
{
    public class SentimentPrediction : SentimentObservation
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
