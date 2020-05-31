using Microsoft.ML.Data;

namespace Experiments.Api.ML.SentimentAnalysis
{
    public class SentimentObservation
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment { get; set; }
    }
}
