using System.Collections;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using Experiments.Api.ML.Clustering;
using Experiments.Api.ML.SentimentAnalysis;
using Microsoft.AspNetCore.Mvc;

namespace Experiments.Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ClusteringController : ControllerBase
    {
        private readonly ClusteringIrisModel model;
        public ClusteringController()
        {
            this.model = new ClusteringIrisModel();
        }

        [HttpGet]
        [Route("predict")]
        public ActionResult<IEnumerable<float>> GetPrediction([FromQuery, Required] float? sepalLength,
                                                              [FromQuery, Required] float? sepalWidth,
                                                              [FromQuery, Required] float? petalLength,
                                                              [FromQuery, Required] float? petalWidth)
        {
            var observation = new IrisObservation
            {
                SepalLength = sepalLength.Value,
                SepalWidth = sepalWidth.Value,
                PetalLength = petalLength.Value,
                PetalWidth = petalWidth.Value
            };

            var prediction = this.model.Predict(observation);

            return prediction.Distances;
        }

        [HttpPost]
        [Route("train")]
        public ActionResult PostTrain()
        {
            var metrics = this.model.Train();

            this.model.Save();

            return this.Ok(metrics);
        }
    }
}
