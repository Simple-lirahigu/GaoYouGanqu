/**
 * 高邮地区 2020 年模型与植被指数对比实验
 *
 * 实验因素：
 * 1. 分类器：RF、CART、SVM；
 * 2. 植被指数月时序：NDVI、SAVI、OSAVI；
 * 3. 验证方式：随机 70/30、1 km 空间分块 70/30。
 *
 * 共 3 × 3 × 2 = 18 组精度实验。
 *
 * 土地利用类别：
 * 0 水体（包含养殖塘）、1 林地、2 耕地、3 建筑、4 其他。
 *
 * 说明：
 * - 随机验证用于和参考论文的 7:3 验证结果进行方法对照；
 * - 空间验证用于评价模型空间泛化能力，应作为论文主要结果；
 * - 本脚本不保证 OA 达到 0.90，最终结果以独立测试指标为准。
 */

// ============================================================================
// 0. 参数
// ============================================================================

var AOI_ASSET = 'projects/ee-yangsimple237/assets/GYBJ';
var LANDUSE_ASSET = 'projects/ee-yangsimple237/assets/2020tudi';
var LANDUSE_BAND = 'b1';

var START_DATE = '2020-01-01';
var END_DATE = '2021-01-01';
var SCALE = 10;
var GRID_SIZE = 1000;
var EXPORT_CRS = 'EPSG:32650';
var RANDOM_SEED = 20200625;

var ORIGINAL_CLASSES = [1, 2, 5, 7, 11];
var MODEL_CLASSES = [0, 1, 2, 3, 4];
var CLASS_NAMES = ['水体_含养殖塘', '林地', '耕地', '建筑', '其他'];
var CLASS_PALETTE = ['419BDF', '397D49', 'E49635', 'C4281B', 'A59B8F'];

// 随机验证：先在全区每类抽样，再按 70/30 随机划分。
var RANDOM_POINTS_PER_CLASS = 600;
var RANDOM_TRAIN_RATIO = 0.70;

// 空间验证：在互斥的 1 km 网格中分别分层抽样。
var SPATIAL_TRAIN_PER_CLASS = 600;
var SPATIAL_TEST_PER_CLASS = 250;

var RF_TREES = 500;
var RF_MTRY = 8;
var RF_MIN_LEAF = 3;
var RF_BAG_FRACTION = 0.7;

var CART_MAX_NODES = 100;
var CART_MIN_LEAF = 5;

var SVM_COST = 10;
var SVM_GAMMA = 0.01;

var DRIVE_FOLDER = 'GEE_Gaoyou_Model_Comparison';

var aoi = ee.FeatureCollection(AOI_ASSET);
var region = aoi.geometry();
var rawLabel = ee.Image(LANDUSE_ASSET).select(LANDUSE_BAND);
var validMask = rawLabel.eq(1)
  .or(rawLabel.eq(2))
  .or(rawLabel.eq(5))
  .or(rawLabel.eq(7))
  .or(rawLabel.eq(11));
var modelLabel = rawLabel
  .remap(ORIGINAL_CLASSES, MODEL_CLASSES)
  .rename('model_label')
  .updateMask(validMask)
  .toInt16();

Map.centerObject(aoi, 10);
Map.addLayer(
  aoi.style({color: 'FF0000', fillColor: '00000000'}),
  {},
  '高邮研究区'
);
Map.addLayer(
  modelLabel,
  {min: 0, max: 4, palette: CLASS_PALETTE},
  '2020土地利用标签',
  false
);

// ============================================================================
// 1. Sentinel-2预处理与指数
// ============================================================================

function maskS2(image) {
  var scl = image.select('SCL');
  var valid = scl.neq(0)
    .and(scl.neq(1))
    .and(scl.neq(3))
    .and(scl.neq(7))
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  return image.updateMask(valid)
    .select(
      ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
      [
        'blue', 'green', 'red', 're1', 're2', 're3',
        'nir', 'nirNarrow', 'swir1', 'swir2'
      ]
    )
    .multiply(0.0001)
    .copyProperties(image, ['system:time_start']);
}

function addIndices(image) {
  var ndvi = image.normalizedDifference(['nir', 'red']).rename('NDVI');
  var savi = image.expression(
    '((nir - red) / (nir + red + L)) * (1 + L)',
    {
      nir: image.select('nir'),
      red: image.select('red'),
      L: 0.5
    }
  ).rename('SAVI');
  var osavi = image.expression(
    '1.16 * (nir - red) / (nir + red + 0.16)',
    {
      nir: image.select('nir'),
      red: image.select('red')
    }
  ).rename('OSAVI');
  var mndwi = image.normalizedDifference(['green', 'swir1']).rename('MNDWI');
  var ndbi = image.normalizedDifference(['swir1', 'nir']).rename('NDBI');
  var ndre1 = image.normalizedDifference(['nirNarrow', 're1']).rename('NDRE1');
  var ndre2 = image.normalizedDifference(['nirNarrow', 're2']).rename('NDRE2');
  var ndre3 = image.normalizedDifference(['nirNarrow', 're3']).rename('NDRE3');

  return image.addBands([
    ndvi, savi, osavi, mndwi, ndbi, ndre1, ndre2, ndre3
  ]);
}

var commonBands = [
  'blue', 'green', 'red', 're1', 're2', 're3',
  'nir', 'nirNarrow', 'swir1', 'swir2',
  'MNDWI', 'NDBI', 'NDRE1', 'NDRE2', 'NDRE3'
];
var allBands = commonBands.concat(['NDVI', 'SAVI', 'OSAVI']);

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
  .map(maskS2)
  .map(addIndices)
  .select(allBands);

var annualMedian = s2.median().select(allBands).unmask(0);

function safeComposite(start, end, bands) {
  var subset = s2.filterDate(start, end);
  return ee.Image(ee.Algorithms.If(
    subset.size().gt(0),
    subset.median().select(bands),
    annualMedian.select(bands)
  )).unmask(annualMedian.select(bands)).unmask(0);
}

function makeQuarter(start, end, prefix) {
  var image = safeComposite(start, end, commonBands);
  return image.rename(commonBands.map(function(name) {
    return prefix + '_' + name;
  }));
}

var commonQuarterFeatures = ee.Image.cat([
  makeQuarter('2020-01-01', '2020-04-01', 'Q1'),
  makeQuarter('2020-04-01', '2020-07-01', 'Q2'),
  makeQuarter('2020-07-01', '2020-10-01', 'Q3'),
  makeQuarter('2020-10-01', '2021-01-01', 'Q4')
]);

function makeMonthlyIndexFeatures(indexName) {
  var images = [];
  for (var month = 1; month <= 12; month++) {
    var start = ee.Date.fromYMD(2020, month, 1);
    var end = start.advance(1, 'month');
    var prefix = month < 10 ? 'M0' + month : 'M' + month;
    images.push(
      safeComposite(start, end, [indexName])
        .rename(prefix + '_' + indexName)
    );
  }

  var stats = s2.select(indexName).reduce(
    ee.Reducer.mean()
      .combine({reducer2: ee.Reducer.min(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
  ).unmask(0).rename([
    indexName + '_mean',
    indexName + '_min',
    indexName + '_max',
    indexName + '_stdDev'
  ]);

  return ee.Image.cat(images).addBands(stats);
}

var ndviFeatures = commonQuarterFeatures
  .addBands(makeMonthlyIndexFeatures('NDVI'))
  .toFloat()
  .clip(region);
var saviFeatures = commonQuarterFeatures
  .addBands(makeMonthlyIndexFeatures('SAVI'))
  .toFloat()
  .clip(region);
var osaviFeatures = commonQuarterFeatures
  .addBands(makeMonthlyIndexFeatures('OSAVI'))
  .toFloat()
  .clip(region);

var featureSets = [
  {name: 'NDVI', image: ndviFeatures},
  {name: 'SAVI', image: saviFeatures},
  {name: 'OSAVI', image: osaviFeatures}
];

print('Sentinel-2影像数量：', s2.size());
print('单套实验特征数量：', ndviFeatures.bandNames().size());

// ============================================================================
// 2. 固定样本位置与两种验证方式
// ============================================================================

var gridProjection = ee.Projection(EXPORT_CRS).atScale(GRID_SIZE);
var sampleProjection = ee.Projection(EXPORT_CRS).atScale(SCALE);
var coordinates = ee.Image.pixelCoordinates(gridProjection);
var gridX = coordinates.select('x').toInt64();
var gridY = coordinates.select('y').toInt64();
var gridId = gridX.multiply(1000000).add(gridY).rename('grid_id').toInt64();
var gridHash = gridX.multiply(73856093)
  .add(gridY.multiply(19349663))
  .add(RANDOM_SEED)
  .abs()
  .mod(100);

var spatialTrainMask = gridHash.lt(70);
var spatialTestMask = gridHash.gte(70);
var labelWithGrid = modelLabel.addBands(gridId);

var randomMasterPoints = labelWithGrid.stratifiedSample({
  numPoints: RANDOM_POINTS_PER_CLASS,
  classBand: 'model_label',
  region: region,
  scale: SCALE,
  projection: sampleProjection,
  seed: RANDOM_SEED + 1,
  dropNulls: true,
  tileScale: 4,
  geometries: true
}).randomColumn('random_key', RANDOM_SEED + 2);

var randomTrainPoints = randomMasterPoints.filter(
  ee.Filter.lt('random_key', RANDOM_TRAIN_RATIO)
);
var randomTestPoints = randomMasterPoints.filter(
  ee.Filter.gte('random_key', RANDOM_TRAIN_RATIO)
);

function stratifiedSpatialPoints(mask, pointsPerClass, seedOffset) {
  return labelWithGrid.updateMask(mask).stratifiedSample({
    numPoints: pointsPerClass,
    classBand: 'model_label',
    region: region,
    scale: SCALE,
    projection: sampleProjection,
    seed: RANDOM_SEED + seedOffset,
    dropNulls: true,
    tileScale: 4,
    geometries: true
  });
}

var spatialTrainPoints = stratifiedSpatialPoints(
  spatialTrainMask,
  SPATIAL_TRAIN_PER_CLASS,
  10
);
var spatialTestPoints = stratifiedSpatialPoints(
  spatialTestMask,
  SPATIAL_TEST_PER_CLASS,
  11
);

var splitImage = ee.Image(0)
  .where(spatialTrainMask, 1)
  .where(spatialTestMask, 2)
  .rename('spatial_split')
  .clip(region);

print('随机训练样本：', randomTrainPoints.aggregate_histogram('model_label'));
print('随机测试样本：', randomTestPoints.aggregate_histogram('model_label'));
print('空间训练样本：', spatialTrainPoints.aggregate_histogram('model_label'));
print('空间测试样本：', spatialTestPoints.aggregate_histogram('model_label'));

Map.addLayer(
  splitImage,
  {min: 1, max: 2, palette: ['4DAF4A', 'E41A1C']},
  '空间验证分区：训练/测试',
  false
);

// ============================================================================
// 3. 模型与指标
// ============================================================================

var algorithmConfigs = [
  {name: 'RF'},
  {name: 'CART'},
  {name: 'SVM'}
];

function buildClassifier(algorithmName) {
  if (algorithmName === 'RF') {
    return ee.Classifier.smileRandomForest({
      numberOfTrees: RF_TREES,
      variablesPerSplit: RF_MTRY,
      minLeafPopulation: RF_MIN_LEAF,
      bagFraction: RF_BAG_FRACTION,
      seed: RANDOM_SEED
    });
  }
  if (algorithmName === 'CART') {
    return ee.Classifier.smileCart({
      maxNodes: CART_MAX_NODES,
      minLeafPopulation: CART_MIN_LEAF
    });
  }
  return ee.Classifier.libsvm({
    svmType: 'C_SVC',
    kernelType: 'RBF',
    shrinking: true,
    gamma: SVM_GAMMA,
    cost: SVM_COST
  });
}

function meanList(values) {
  return ee.Number(ee.List(values).reduce(ee.Reducer.mean()));
}

function metricsFromSamples(samples) {
  var matrix = samples.errorMatrix(
    'model_label',
    'classification',
    MODEL_CLASSES
  );
  var producer = ee.Array(matrix.producersAccuracy()).toList().flatten();
  var user = ee.Array(matrix.consumersAccuracy()).toList().flatten();
  var f1 = ee.List.sequence(0, MODEL_CLASSES.length - 1).map(function(index) {
    index = ee.Number(index);
    var recall = ee.Number(producer.get(index));
    var precision = ee.Number(user.get(index));
    return ee.Number(ee.Algorithms.If(
      recall.add(precision).gt(0),
      recall.multiply(precision).multiply(2).divide(
        recall.add(precision)
      ),
      0
    ));
  });

  return ee.Dictionary({
    matrix: matrix.array().toList(),
    overall_accuracy: matrix.accuracy(),
    kappa: matrix.kappa(),
    producer_accuracy: producer,
    user_accuracy: user,
    f1: f1,
    macro_f1: meanList(f1)
  });
}

var summaryRows = [];
var classRows = [];
var matrixRows = [];
var spatialPredictionBands = [];
var paperCandidateClassification = null;

function evaluateExperiment(
  featureSet,
  algorithmName,
  validationName,
  trainPoints,
  testPoints
) {
  var featureNames = featureSet.image.bandNames();
  var trainSamples = featureSet.image.sampleRegions({
    collection: trainPoints,
    properties: ['model_label', 'grid_id'],
    scale: SCALE,
    projection: sampleProjection,
    tileScale: 4,
    geometries: false
  });
  var testSamples = featureSet.image.sampleRegions({
    collection: testPoints,
    properties: ['model_label', 'grid_id'],
    scale: SCALE,
    projection: sampleProjection,
    tileScale: 4,
    geometries: false
  });

  var classifier = buildClassifier(algorithmName).train({
    features: trainSamples,
    classProperty: 'model_label',
    inputProperties: featureNames
  });
  var assessed = testSamples.classify(classifier);
  var metrics = metricsFromSamples(assessed);
  var producer = ee.List(metrics.get('producer_accuracy'));
  var user = ee.List(metrics.get('user_accuracy'));
  var f1 = ee.List(metrics.get('f1'));
  var experimentId = algorithmName + '_' + featureSet.name + '_' + validationName;

  summaryRows.push(ee.Feature(null, {
    record_type: 'experiment_summary',
    experiment_id: experimentId,
    algorithm: algorithmName,
    index_set: featureSet.name,
    validation: validationName,
    train_sample_count: trainSamples.size(),
    test_sample_count: testSamples.size(),
    overall_accuracy: metrics.get('overall_accuracy'),
    kappa: metrics.get('kappa'),
    macro_f1: metrics.get('macro_f1')
  }));

  for (var classIndex = 0; classIndex < MODEL_CLASSES.length; classIndex++) {
    classRows.push(ee.Feature(null, {
      record_type: 'class_metric',
      experiment_id: experimentId,
      algorithm: algorithmName,
      index_set: featureSet.name,
      validation: validationName,
      model_code: MODEL_CLASSES[classIndex],
      original_code: ORIGINAL_CLASSES[classIndex],
      class_name: CLASS_NAMES[classIndex],
      precision: user.get(classIndex),
      recall: producer.get(classIndex),
      f1: f1.get(classIndex)
    }));
  }

  var matrix = ee.List(metrics.get('matrix'));
  for (var actualIndex = 0; actualIndex < MODEL_CLASSES.length; actualIndex++) {
    for (
      var predictedIndex = 0;
      predictedIndex < MODEL_CLASSES.length;
      predictedIndex++
    ) {
      matrixRows.push(ee.Feature(null, {
        record_type: 'confusion_matrix',
        experiment_id: experimentId,
        algorithm: algorithmName,
        index_set: featureSet.name,
        validation: validationName,
        actual_code: MODEL_CLASSES[actualIndex],
        actual_name: CLASS_NAMES[actualIndex],
        predicted_code: MODEL_CLASSES[predictedIndex],
        predicted_name: CLASS_NAMES[predictedIndex],
        sample_count: ee.List(matrix.get(actualIndex)).get(predictedIndex)
      }));
    }
  }

  if (validationName === 'spatial') {
    var prediction = featureSet.image
      .classify(classifier)
      .rename(algorithmName + '_' + featureSet.name)
      .toInt16();
    spatialPredictionBands.push(prediction);

    // 与参考论文相对应的候选结果：SAVI时序 + RF。
    if (algorithmName === 'RF' && featureSet.name === 'SAVI') {
      paperCandidateClassification = prediction;
    }
  }
}

for (var featureIndex = 0; featureIndex < featureSets.length; featureIndex++) {
  for (
    var algorithmIndex = 0;
    algorithmIndex < algorithmConfigs.length;
    algorithmIndex++
  ) {
    var currentFeatureSet = featureSets[featureIndex];
    var currentAlgorithm = algorithmConfigs[algorithmIndex].name;

    evaluateExperiment(
      currentFeatureSet,
      currentAlgorithm,
      'random',
      randomTrainPoints,
      randomTestPoints
    );
    evaluateExperiment(
      currentFeatureSet,
      currentAlgorithm,
      'spatial',
      spatialTrainPoints,
      spatialTestPoints
    );
  }
}

var summaryTable = ee.FeatureCollection(summaryRows);
var classMetricTable = ee.FeatureCollection(classRows);
var confusionTable = ee.FeatureCollection(matrixRows);
var completeAccuracyTable = summaryTable
  .merge(classMetricTable)
  .merge(confusionTable);

print(
  '模型汇总结果（按Macro-F1降序）：',
  summaryTable.sort('macro_f1', false)
);
print('各类别Precision、Recall和F1：', classMetricTable);

// ============================================================================
// 4. 地图显示与导出
// ============================================================================

var spatialPredictionStack = ee.Image.cat(spatialPredictionBands);
var paperCandidateDisplay = ee.Image(paperCandidateClassification);

Map.addLayer(
  paperCandidateDisplay,
  {min: 0, max: 4, palette: CLASS_PALETTE},
  '论文参考候选：RF + SAVI + 空间验证',
  true
);
Map.addLayer(
  makeMonthlyIndexFeatures('NDVI'),
  {},
  'NDVI月时序特征',
  false
);
Map.addLayer(
  makeMonthlyIndexFeatures('SAVI'),
  {},
  'SAVI月时序特征',
  false
);
Map.addLayer(
  makeMonthlyIndexFeatures('OSAVI'),
  {},
  'OSAVI月时序特征',
  false
);

Export.table.toDrive({
  collection: completeAccuracyTable,
  description: 'Gaoyou_model_index_comparison_accuracy_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_model_index_comparison_accuracy_2020',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: summaryTable.sort('macro_f1', false),
  description: 'Gaoyou_model_index_comparison_summary_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_model_index_comparison_summary_2020',
  fileFormat: 'CSV'
});

Export.image.toDrive({
  image: spatialPredictionStack.clip(region),
  description: 'Gaoyou_nine_spatial_model_predictions_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_nine_spatial_model_predictions_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Export.image.toDrive({
  image: paperCandidateDisplay.clip(region),
  description: 'Gaoyou_RF_SAVI_spatial_prediction_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_rf_savi_spatial_prediction_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

