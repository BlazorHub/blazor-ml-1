using Microsoft.ML.Data;

namespace Experiments.Api.ML.Clustering
{
    public class IrisPrediction : IrisObservation
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId { get; set; }

        [ColumnName("Score")]
        public float[] Distances { get; set; }
    }
}
