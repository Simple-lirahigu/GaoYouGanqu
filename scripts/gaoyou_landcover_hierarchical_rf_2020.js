/**
 * 高邮地区 2020 年增强版分层随机森林土地利用分类
 *
 * 第一层：永久水体 / 非永久水体二分类。
 * 第二层：在非水体区域识别林地、耕地、建筑和其他。
 *
 * 主要改进：
 * 1. 原始类别重映射为连续编码 0~4；
 * 2. 使用月尺度 NDVI/MNDWI/NDWI/EVI 与 Sentinel-1 时序；
 * 3. 加入 JRC Global Surface Water 和 Dynamic World 概率特征；
 * 4. 训练标签向类别内部腐蚀 1 个像元，减少边界混合样本；
 * 5. 水体概率阈值由调参区确定，并惩罚“耕地误判为水体”；
 * 6. 独立测试区使用未经腐蚀的原始标签。
 *
 * 精度含义：
 * 精度表示模型对 2020tudi 标签的空间泛化能力，不是独立地面真实性精度。
 */

// ============================================================================
// 0. 参数
// ============================================================================

var AOI_ASSET = 'projects/ee-yangsimple237/assets/GYBJ';
var LANDUSE_ASSET = 'projects/ee-yangsimple237/assets/2020tudi';
// 将 waterSample_shp 中修正后的 SHP 上传到此 GEE 表资产。
var REVIEWED_WATER_ASSET =
  'projects/ee-yangsimple237/assets/gaoyou_water_samples_reviewed_2020';
var LANDUSE_BAND = 'b1';

var START_DATE = '2020-01-01';
var END_DATE = '2021-01-01';
var SCALE = 10;
var GRID_SIZE = 1000;
var EXPORT_CRS = 'EPSG:32650';
var RANDOM_SEED = 20200624;

// 原始编码 -> 连续模型编码。
// 0水体、1林地、2耕地、3建筑、4其他。
var ORIGINAL_CLASSES = [1, 2, 5, 7, 11];
var MODEL_CLASSES = [0, 1, 2, 3, 4];
var CLASS_NAMES = ['水体', '林地', '耕地', '建筑', '其他'];
var CLASS_PALETTE = ['419BDF', '397D49', 'E49635', 'C4281B', 'A59B8F'];
var WATER_MODEL_CODE = 0;
var CROPLAND_MODEL_CODE = 2;

var WATER_TRAIN_PER_CLASS = 1500;
var WATER_TUNE_PER_CLASS = 400;
var NONWATER_TRAIN_PER_CLASS = 1000;
var NONWATER_TUNE_PER_CLASS = 300;
var TEST_PER_CLASS = 300;

var N_TREES = 500;
var BAG_FRACTION = 0.7;
var LABEL_EROSION_PIXELS = 1;
var WATER_THRESHOLDS = [0.45, 0.55, 0.65, 0.75];
var DRIVE_FOLDER = 'GEE_Gaoyou_Hierarchical_RF';

var aoi = ee.FeatureCollection(AOI_ASSET);
var region = aoi.geometry();
var rawLabel = ee.Image(LANDUSE_ASSET).select(LANDUSE_BAND).rename('original_label');

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

// 仅清洗训练/调参标签；测试标签保持原始边界。
var cleanLabelImages = MODEL_CLASSES.map(function(code) {
  var classInterior = modelLabel.eq(code)
    .focalMin(LABEL_EROSION_PIXELS, 'square', 'pixels', 1);
  return ee.Image.constant(code)
    .updateMask(classInterior)
    .rename('model_label')
    .toInt16();
});
var cleanModelLabel = ee.ImageCollection.fromImages(cleanLabelImages)
  .mosaic()
  .rename('model_label');

Map.centerObject(aoi, 10);
Map.addLayer(
  aoi.style({color: 'FF0000', fillColor: '00000000'}),
  {},
  '高邮研究区'
);
Map.addLayer(
  modelLabel,
  {min: 0, max: 4, palette: CLASS_PALETTE},
  '原始五类标签',
  false
);
Map.addLayer(
  cleanModelLabel,
  {min: 0, max: 4, palette: CLASS_PALETTE},
  '训练标签内部区域',
  false
);

print('原始类别编码：', ORIGINAL_CLASSES);
print('模型连续编码：', MODEL_CLASSES);
print('类别名称：', CLASS_NAMES);

// ============================================================================
// 1. Sentinel-2 月尺度和季度特征
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
      ['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B8A', 'B11', 'B12'],
      ['blue', 'green', 'red', 're1', 're2', 'nir', 'nirNarrow', 'swir1', 'swir2']
    )
    .multiply(0.0001)
    .copyProperties(image, ['system:time_start']);
}

function addS2Indices(image) {
  var ndvi = image.normalizedDifference(['nir', 'red']).rename('NDVI');
  var evi = image.expression(
    '2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)',
    {
      nir: image.select('nir'),
      red: image.select('red'),
      blue: image.select('blue')
    }
  ).rename('EVI');
  var ndwi = image.normalizedDifference(['green', 'nir']).rename('NDWI');
  var mndwi = image.normalizedDifference(['green', 'swir1']).rename('MNDWI');
  var ndbi = image.normalizedDifference(['swir1', 'nir']).rename('NDBI');
  var bsi = image.expression(
    '((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))',
    {
      swir1: image.select('swir1'),
      red: image.select('red'),
      nir: image.select('nir'),
      blue: image.select('blue')
    }
  ).rename('BSI');
  var ndre = image.normalizedDifference(['nirNarrow', 're1']).rename('NDRE');
  return image.addBands([ndvi, evi, ndwi, mndwi, ndbi, bsi, ndre]);
}

var s2BaseBands = [
  'blue', 'green', 'red', 're1', 're2', 'nir', 'nirNarrow', 'swir1', 'swir2',
  'NDVI', 'EVI', 'NDWI', 'MNDWI', 'NDBI', 'BSI', 'NDRE'
];
var waterIndexBands = ['NDVI', 'EVI', 'NDWI', 'MNDWI'];

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
  .map(maskS2)
  .map(addS2Indices)
  .select(s2BaseBands);

var s2Annual = s2.median().select(s2BaseBands);

function safeS2Composite(start, end, bands) {
  var subset = s2.filterDate(start, end);
  return ee.Image(ee.Algorithms.If(
    subset.size().gt(0),
    subset.median().select(bands),
    s2Annual.select(bands)
  )).unmask(s2Annual.select(bands)).unmask(0);
}

function makeQuarter(start, end, prefix) {
  var image = safeS2Composite(start, end, s2BaseBands);
  return image.rename(s2BaseBands.map(function(name) {
    return prefix + '_' + name;
  }));
}

var s2QuarterFeatures = ee.Image.cat([
  makeQuarter('2020-01-01', '2020-04-01', 'S2_Q1'),
  makeQuarter('2020-04-01', '2020-07-01', 'S2_Q2'),
  makeQuarter('2020-07-01', '2020-10-01', 'S2_Q3'),
  makeQuarter('2020-10-01', '2021-01-01', 'S2_Q4')
]);

var monthlyS2Images = [];
var monthlyMndwiMasks = [];
var monthlyVegetationMasks = [];

for (var month = 1; month <= 12; month++) {
  var monthStart = ee.Date.fromYMD(2020, month, 1);
  var monthEnd = monthStart.advance(1, 'month');
  var prefix = 'M' + (month < 10 ? '0' + month : month);
  var monthly = safeS2Composite(monthStart, monthEnd, waterIndexBands);

  monthlyS2Images.push(monthly.rename(waterIndexBands.map(function(name) {
    return prefix + '_' + name;
  })));
  monthlyMndwiMasks.push(monthly.select('MNDWI').gt(0));
  monthlyVegetationMasks.push(monthly.select('NDVI').gt(0.4));
}

var s2MonthlyFeatures = ee.Image.cat(monthlyS2Images);
var waterMonthCount = ee.ImageCollection.fromImages(monthlyMndwiMasks)
  .sum()
  .rename('S2_water_month_count');
var vegetationMonthCount = ee.ImageCollection.fromImages(monthlyVegetationMasks)
  .sum()
  .rename('S2_vegetation_month_count');

var s2StatsReducer = ee.Reducer.mean()
  .combine({reducer2: ee.Reducer.min(), sharedInputs: true})
  .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
  .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true});
var s2AnnualStats = s2.select(waterIndexBands)
  .reduce(s2StatsReducer)
  .unmask(0);

print('2020年 Sentinel-2 影像数：', s2.size());

// ============================================================================
// 2. Sentinel-1 月尺度与季度特征
// ============================================================================

function prepareS1(image) {
  var vv = image.select('VV').rename('VV');
  var vh = image.select('VH').rename('VH');
  return vv.addBands(vh)
    .addBands(vv.subtract(vh).rename('VV_minus_VH'))
    .copyProperties(image, ['system:time_start']);
}

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(region)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH'])
  .map(prepareS1);

var s1Bands = ['VV', 'VH', 'VV_minus_VH'];
var s1AnnualMedian = ee.Image(ee.Algorithms.If(
  s1.size().gt(0),
  s1.median().select(s1Bands),
  ee.Image.constant([0, 0, 0]).rename(s1Bands)
)).unmask(0);

function safeS1Median(start, end, bands) {
  var subset = s1.filterDate(start, end);
  return ee.Image(ee.Algorithms.If(
    subset.size().gt(0),
    subset.median().select(bands),
    s1AnnualMedian.select(bands)
  )).unmask(s1AnnualMedian.select(bands)).unmask(0);
}

var monthlyS1Images = [];
for (var s1Month = 1; s1Month <= 12; s1Month++) {
  var s1Start = ee.Date.fromYMD(2020, s1Month, 1);
  var s1End = s1Start.advance(1, 'month');
  var s1Prefix = 'M' + (s1Month < 10 ? '0' + s1Month : s1Month);
  var s1Monthly = safeS1Median(s1Start, s1End, ['VV', 'VH']);
  monthlyS1Images.push(s1Monthly.rename([
    s1Prefix + '_VV',
    s1Prefix + '_VH'
  ]));
}
var s1MonthlyFeatures = ee.Image.cat(monthlyS1Images);

var s1QuarterReducer = ee.Reducer.median()
  .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true});
var s1ReducedNames = [
  'VV_median', 'VH_median', 'VV_minus_VH_median',
  'VV_stdDev', 'VH_stdDev', 'VV_minus_VH_stdDev'
];
var s1AnnualReduced = ee.Image(ee.Algorithms.If(
  s1.size().gt(0),
  s1.reduce(s1QuarterReducer).select(s1ReducedNames),
  ee.Image.constant([0, 0, 0, 0, 0, 0]).rename(s1ReducedNames)
)).unmask(0);

function makeS1Quarter(start, end, prefix) {
  var subset = s1.filterDate(start, end);
  var reduced = ee.Image(ee.Algorithms.If(
    subset.size().gt(0),
    subset.reduce(s1QuarterReducer).select(s1ReducedNames),
    s1AnnualReduced
  )).unmask(s1AnnualReduced).unmask(0);
  return reduced.rename(s1ReducedNames.map(function(name) {
    return prefix + '_' + name;
  }));
}

var s1QuarterFeatures = ee.Image.cat([
  makeS1Quarter('2020-01-01', '2020-04-01', 'S1_Q1'),
  makeS1Quarter('2020-04-01', '2020-07-01', 'S1_Q2'),
  makeS1Quarter('2020-07-01', '2020-10-01', 'S1_Q3'),
  makeS1Quarter('2020-10-01', '2021-01-01', 'S1_Q4')
]);

var s1StatsReducer = ee.Reducer.mean()
  .combine({reducer2: ee.Reducer.min(), sharedInputs: true})
  .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
  .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true});
var s1AnnualStats = s1.select(['VV', 'VH'])
  .reduce(s1StatsReducer)
  .unmask(0);

print('2020年 Sentinel-1 影像数（升轨+降轨）：', s1.size());

// ============================================================================
// 3. JRC和Dynamic World先验特征
// ============================================================================

var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select(['occurrence', 'seasonality', 'recurrence', 'max_extent'])
  .unmask(0)
  .rename([
    'JRC_occurrence',
    'JRC_seasonality',
    'JRC_recurrence',
    'JRC_max_extent'
  ]);

var dynamicWorld = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(region)
  .filterDate(START_DATE, END_DATE)
  .select(['water', 'trees', 'crops', 'built', 'flooded_vegetation'])
  .mean()
  .unmask(0)
  .rename([
    'DW_water',
    'DW_trees',
    'DW_crops',
    'DW_built',
    'DW_flooded_vegetation'
  ]);

// 水体阶段强调月尺度持续性。
var waterPredictors = s2MonthlyFeatures
  .addBands(s2AnnualStats)
  .addBands(waterMonthCount)
  .addBands(vegetationMonthCount)
  .addBands(s1MonthlyFeatures)
  .addBands(s1AnnualStats)
  .addBands(jrc)
  .addBands(dynamicWorld)
  .toFloat()
  .unmask(0)
  .clip(region);

// 非水体阶段保留季度完整光谱和雷达特征。
var nonwaterPredictors = s2QuarterFeatures
  .addBands(s2AnnualStats)
  .addBands(s1QuarterFeatures)
  .addBands(jrc)
  .addBands(dynamicWorld)
  .toFloat()
  .unmask(0)
  .clip(region);

print('水体阶段特征数：', waterPredictors.bandNames().size());
print('非水体阶段特征数：', nonwaterPredictors.bandNames().size());

// ============================================================================
// 4. 1 km空间分区
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

var trainMask = gridHash.lt(60);
var tuneMask = gridHash.gte(60).and(gridHash.lt(80));
var testMask = gridHash.gte(80);
var splitImage = ee.Image(0)
  .where(trainMask, 1)
  .where(tuneMask, 2)
  .where(testMask, 3)
  .rename('spatial_split')
  .clip(region);

Map.addLayer(
  splitImage,
  {min: 1, max: 3, palette: ['4DAF4A', '377EB8', 'E41A1C']},
  '训练/调参/测试空间分区',
  false
);

// ============================================================================
// 5. 样本抽取
// ============================================================================

// 人工修正水体样本：
// samp_type：1训练、2调参、3独立测试；
// man_label：0非水体、1水体；
// qa_stat：1表示已经人工核查。
// 脚本只筛选有效记录，不修改或删除原始资产中的任何点。
var reviewedWaterRaw = ee.FeatureCollection(REVIEWED_WATER_ASSET)
  .filter(ee.Filter.eq('qa_stat', 1))
  .filter(ee.Filter.inList('samp_type', [1, 2, 3]))
  .filter(ee.Filter.inList('man_label', [0, 1]));

function prepareReviewedWaterSamples(sampleType, splitName) {
  var reviewedSubset = reviewedWaterRaw.filter(
    ee.Filter.eq('samp_type', sampleType)
  );
  return waterPredictors.sampleRegions({
    collection: reviewedSubset,
    properties: [
      'samp_type', 'man_label', 'qa_stat', 'grid_id',
      'qa_note', 'reviewer', 'rev_date'
    ],
    scale: SCALE,
    projection: sampleProjection,
    tileScale: 4,
    geometries: true
  }).map(function(feature) {
    var waterLabel = ee.Number(feature.get('man_label')).int();
    return feature.set({
      water_label: waterLabel,
      // 人工非水体点只用于水体二分类，未知其具体四分类类别。
      model_label: ee.Algorithms.If(waterLabel.eq(1), WATER_MODEL_CODE, -1),
      sample_split: splitName
    });
  });
}

var waterTrain = prepareReviewedWaterSamples(1, 'reviewed_water_train');
var waterTune = prepareReviewedWaterSamples(2, 'reviewed_water_tune');
var reviewedWaterTest = prepareReviewedWaterSamples(
  3,
  'reviewed_water_independent_test'
);

var nonwaterLabel = cleanModelLabel
  .updateMask(cleanModelLabel.neq(WATER_MODEL_CODE))
  .rename('nonwater_label');
var nonwaterSampleSource = nonwaterPredictors
  .addBands(nonwaterLabel)
  .addBands(gridId);

function sampleNonwater(splitMaskValue, pointsPerClass, splitName, seedOffset) {
  return nonwaterSampleSource.updateMask(splitMaskValue).stratifiedSample({
    numPoints: 0,
    classBand: 'nonwater_label',
    classValues: [1, 2, 3, 4],
    classPoints: [
      pointsPerClass,
      pointsPerClass,
      pointsPerClass,
      pointsPerClass
    ],
    region: region,
    scale: SCALE,
    projection: sampleProjection,
    seed: RANDOM_SEED + seedOffset,
    dropNulls: true,
    tileScale: 4,
    geometries: true
  }).map(function(feature) {
    return feature.set('sample_split', splitName);
  });
}

var nonwaterTrain = sampleNonwater(
  trainMask,
  NONWATER_TRAIN_PER_CLASS,
  'nonwater_train',
  20
);
var nonwaterTune = sampleNonwater(
  tuneMask,
  NONWATER_TUNE_PER_CLASS,
  'nonwater_tune',
  21
);

// 测试集使用未经腐蚀的原始标签。
var rawTestPoints = modelLabel.addBands(gridId)
  .updateMask(testMask)
  .stratifiedSample({
    numPoints: 0,
    classBand: 'model_label',
    classValues: MODEL_CLASSES,
    classPoints: [
      TEST_PER_CLASS,
      TEST_PER_CLASS,
      TEST_PER_CLASS,
      TEST_PER_CLASS,
      TEST_PER_CLASS
    ],
    region: region,
    scale: SCALE,
    projection: sampleProjection,
    seed: RANDOM_SEED + 30,
    dropNulls: true,
    tileScale: 4,
    geometries: true
  }).map(function(feature) {
    return feature.set('sample_split', 'independent_test');
  });

print('人工修正水体训练样本：', waterTrain.aggregate_histogram('water_label'));
print('人工修正水体调参样本：', waterTune.aggregate_histogram('water_label'));
print(
  '人工修正水体独立测试样本：',
  reviewedWaterTest.aggregate_histogram('water_label')
);
print('非水体训练样本：', nonwaterTrain.aggregate_histogram('nonwater_label'));
print('非水体调参样本：', nonwaterTune.aggregate_histogram('nonwater_label'));
print('独立测试样本：', rawTestPoints.aggregate_histogram('model_label'));

// ============================================================================
// 6. 指标工具
// ============================================================================

function meanList(values) {
  return ee.Number(ee.List(values).reduce(ee.Reducer.mean()));
}

function multiclassMetrics(samples, actualProperty, predictedProperty, order) {
  var matrix = samples.errorMatrix(actualProperty, predictedProperty, order);
  var producer = ee.Array(matrix.producersAccuracy()).toList().flatten();
  var user = ee.Array(matrix.consumersAccuracy()).toList().flatten();
  var f1 = ee.List.sequence(0, order.length - 1).map(function(index) {
    index = ee.Number(index);
    var recall = ee.Number(producer.get(index));
    var precision = ee.Number(user.get(index));
    return ee.Number(ee.Algorithms.If(
      recall.add(precision).gt(0),
      recall.multiply(precision).multiply(2).divide(recall.add(precision)),
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
    macro_f1: meanList(f1),
    macro_recall: meanList(producer)
  });
}

function probabilityOfClassOne(classifier, predictors) {
  return predictors
    .classify(classifier.setOutputMode('MULTIPROBABILITY'))
    .arrayGet([1])
    .rename('water_probability');
}

// ============================================================================
// 7. 第一层：水体模型和阈值联合调参
// ============================================================================

var waterFeatureNames = waterPredictors.bandNames();
var waterCandidates = [
  {id: 1, mtry: 10, minLeaf: 1},
  {id: 2, mtry: 10, minLeaf: 5},
  {id: 3, mtry: 35, minLeaf: 1},
  {id: 4, mtry: 35, minLeaf: 5}
];

var waterTuneRows = [];

waterCandidates.forEach(function(config) {
  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: N_TREES,
    variablesPerSplit: config.mtry,
    minLeafPopulation: config.minLeaf,
    bagFraction: BAG_FRACTION,
    seed: RANDOM_SEED
  }).train({
    features: waterTrain,
    classProperty: 'water_label',
    inputProperties: waterFeatureNames
  });

  var probability = probabilityOfClassOne(classifier, waterPredictors);
  var tuneWithProbability = probability.sampleRegions({
    collection: waterTune,
    properties: ['water_label', 'model_label', 'grid_id', 'samp_type'],
    scale: SCALE,
    projection: sampleProjection,
    tileScale: 4,
    geometries: false
  });

  WATER_THRESHOLDS.forEach(function(threshold) {
    var assessed = tuneWithProbability.map(function(feature) {
      return feature.set(
        'water_prediction',
        ee.Number(feature.get('water_probability')).gte(threshold).int()
      );
    });
    var metrics = multiclassMetrics(
      assessed,
      'water_label',
      'water_prediction',
      [0, 1]
    );
    var f1 = ee.List(metrics.get('f1'));
    var waterF1 = ee.Number(f1.get(1));

    var reviewedNonwaterRows = assessed.filter(
      ee.Filter.eq('water_label', 0)
    );
    var reviewedFalseWater = reviewedNonwaterRows.filter(
      ee.Filter.eq('water_prediction', 1)
    ).size();
    var reviewedNonwaterFpr = ee.Number(ee.Algorithms.If(
      reviewedNonwaterRows.size().gt(0),
      ee.Number(reviewedFalseWater).divide(reviewedNonwaterRows.size()),
      0
    ));

    // 水体F1优先，同时惩罚人工确认非水体被误判为水体。
    var selectionScore = waterF1
      .subtract(reviewedNonwaterFpr.multiply(0.75))
      .add(ee.Number(metrics.get('overall_accuracy')).multiply(0.01));

    waterTuneRows.push(ee.Feature(null, {
      record_type: 'water_tuning',
      candidate_id: config.id,
      variablesPerSplit: config.mtry,
      minLeafPopulation: config.minLeaf,
      threshold: threshold,
      water_f1: waterF1,
      reviewed_nonwater_fpr: reviewedNonwaterFpr,
      overall_accuracy: metrics.get('overall_accuracy'),
      kappa: metrics.get('kappa'),
      selection_score: selectionScore
    }));
  });
});

var waterTuningTable = ee.FeatureCollection(waterTuneRows)
  .sort('selection_score', false);
var bestWater = ee.Feature(waterTuningTable.first());
print('水体模型调参结果：', waterTuningTable);
print('最优水体模型和阈值：', bestWater);

var finalWaterTrain = waterTrain.merge(waterTune);
var finalWaterClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: N_TREES,
  variablesPerSplit: ee.Number(bestWater.get('variablesPerSplit')),
  minLeafPopulation: ee.Number(bestWater.get('minLeafPopulation')),
  bagFraction: BAG_FRACTION,
  seed: RANDOM_SEED
}).train({
  features: finalWaterTrain,
  classProperty: 'water_label',
  inputProperties: waterFeatureNames
});

var finalWaterProbability = probabilityOfClassOne(
  finalWaterClassifier,
  waterPredictors
);
var bestWaterThreshold = ee.Number(bestWater.get('threshold'));
var finalWaterMask = finalWaterProbability.gte(bestWaterThreshold);
// 最终二值水体预测：1=水体，0=非水体。
var finalWaterPrediction = finalWaterMask
  .unmask(0)
  .rename('water_prediction')
  .toByte();

// ============================================================================
// 7.1 水体样本人工质检数据
// ============================================================================

// 仅导出人工判断水体真伪所需的核心字段，避免把全部月尺度特征写入文件。
var waterQaImage = finalWaterProbability
  .addBands(jrc)
  .addBands(dynamicWorld)
  .addBands(waterMonthCount)
  .addBands(vegetationMonthCount)
  .addBands(s2AnnualStats.select([
    'NDVI_mean', 'NDVI_max', 'NDVI_stdDev',
    'MNDWI_mean', 'MNDWI_max', 'MNDWI_stdDev'
  ]))
  .addBands(s1AnnualStats.select([
    'VV_mean', 'VV_min', 'VV_max', 'VV_stdDev',
    'VH_mean', 'VH_min', 'VH_max', 'VH_stdDev'
  ]));

var waterQaProperties = [
  'sample_split',
  'grid_id',
  'model_label',
  'water_label'
];

function prepareWaterQaSamples(points, sourceName, sampleType) {
  var sampled = waterQaImage.sampleRegions({
    collection: points,
    properties: waterQaProperties,
    scale: SCALE,
    projection: sampleProjection,
    tileScale: 4,
    geometries: true
  });

  return sampled.map(function(feature) {
    var coordinates = feature.geometry().coordinates();
    var jrcOccurrence = ee.Number(feature.get('JRC_occurrence'));
    var dwWater = ee.Number(feature.get('DW_water'));
    var waterMonths = ee.Number(feature.get('S2_water_month_count'));
    var vegetationMonths = ee.Number(feature.get('S2_vegetation_month_count'));
    var waterProbability = ee.Number(feature.get('water_probability'));

    // 仅作人工检查提示：历史水体、年度模型和月度持续性证据均较弱，
    // 或植被月份较多时，优先核查。不能据此自动删除样本。
    var weakWaterEvidence = jrcOccurrence.lt(25)
      .and(dwWater.lt(0.40))
      .and(waterMonths.lte(2));
    var cropLikeEvidence = vegetationMonths.gte(3)
      .and(waterProbability.lt(0.75));
    var qaSuspect = weakWaterEvidence.or(cropLikeEvidence);

    return feature.set({
      qa_source: sourceName,
      // 1=训练点，2=调参点，3=独立测试点。
      sample_type: sampleType,
      sample_type_name: sourceName,
      longitude: coordinates.get(0),
      latitude: coordinates.get(1),
      original_class_code: 1,
      class_name: '水体',
      selected_water_threshold: bestWaterThreshold,
      qa_suspect: ee.Number(ee.Algorithms.If(qaSuspect, 1, 0)),
      qa_reason: ee.String(ee.Algorithms.If(
        weakWaterEvidence,
        'weak_persistent_water_evidence',
        ee.Algorithms.If(
          cropLikeEvidence,
          'seasonal_vegetation_or_cropland_signal',
          'no_automatic_warning'
        )
      ))
    });
  });
}

var waterTrainQa = prepareWaterQaSamples(
  waterTrain.filter(ee.Filter.eq('water_label', 1)),
  'water_training',
  1
);
var waterTuneQa = prepareWaterQaSamples(
  waterTune.filter(ee.Filter.eq('water_label', 1)),
  'water_tuning',
  2
);
var waterTestQa = prepareWaterQaSamples(
  reviewedWaterTest.filter(ee.Filter.eq('water_label', 1)),
  'water_independent_test',
  3
);
var allWaterQaSamples = waterTrainQa
  .merge(waterTuneQa)
  .merge(waterTestQa);

// Shapefile 的 DBF 字段名通常最多 10 个字符。
// 这里构建只含短字段名的质检图层，避免字段被截断后重名。
// 人工核查字段约定：
// man_label：-1=待核查，0=非水体，1=确认水体，2=不确定；
// qa_stat：0=未核查，1=已核查。
function prepareWaterShpSamples(samples) {
  return samples.map(function(feature) {
    return ee.Feature(feature.geometry(), {
      samp_type: feature.get('sample_type'),
      samp_name: feature.get('sample_type_name'),
      orig_cls: feature.get('original_class_code'),
      mdl_cls: feature.get('model_label'),
      wat_lbl: feature.get('water_label'),
      grid_id: feature.get('grid_id'),
      lon: feature.get('longitude'),
      lat: feature.get('latitude'),
      wat_prob: feature.get('water_probability'),
      wat_thr: feature.get('selected_water_threshold'),
      jrc_occ: feature.get('JRC_occurrence'),
      jrc_seas: feature.get('JRC_seasonality'),
      jrc_rec: feature.get('JRC_recurrence'),
      jrc_max: feature.get('JRC_max_extent'),
      dw_water: feature.get('DW_water'),
      dw_crops: feature.get('DW_crops'),
      dw_flood: feature.get('DW_flooded_vegetation'),
      wat_mon: feature.get('S2_water_month_count'),
      veg_mon: feature.get('S2_vegetation_month_count'),
      ndvi_avg: feature.get('NDVI_mean'),
      ndvi_max: feature.get('NDVI_max'),
      mndwi_avg: feature.get('MNDWI_mean'),
      mndwi_max: feature.get('MNDWI_max'),
      vv_avg: feature.get('VV_mean'),
      vh_avg: feature.get('VH_mean'),
      qa_susp: feature.get('qa_suspect'),
      qa_reason: feature.get('qa_reason'),
      man_label: -1,
      qa_stat: 0,
      qa_note: '',
      reviewer: '',
      rev_date: ''
    });
  });
}

var waterTrainShp = prepareWaterShpSamples(waterTrainQa);
var waterTuneShp = prepareWaterShpSamples(waterTuneQa);
var waterTestShp = prepareWaterShpSamples(waterTestQa);
var allWaterShp = waterTrainShp
  .merge(waterTuneShp)
  .merge(waterTestShp);

// 将样本点栅格化到10 m输出网格：
// 0=无样本，1=训练水体点，2=调参水体点，3=独立测试水体点。
var waterSampleTypeRaster = allWaterQaSamples
  .reduceToImage(['sample_type'], ee.Reducer.first())
  .rename('water_sample_type')
  .unmask(0)
  .toByte()
  .clip(region);

print('人工质检水体训练点数量：', waterTrainQa.size());
print('人工质检水体调参点数量：', waterTuneQa.size());
print('人工质检独立测试水体点数量：', waterTestQa.size());
print(
  '自动提示需优先核查的水体点数量：',
  allWaterQaSamples.filter(ee.Filter.eq('qa_suspect', 1)).size()
);

Map.addLayer(
  waterTrainQa.style({color: '00FFFF', pointSize: 4}),
  {},
  '人工质检-水体训练点',
  false
);
Map.addLayer(
  waterTuneQa.style({color: '0055FF', pointSize: 4}),
  {},
  '人工质检-水体调参点',
  false
);
Map.addLayer(
  waterTestQa.style({color: 'FF00FF', pointSize: 4}),
  {},
  '人工质检-独立测试水体点',
  false
);
Map.addLayer(
  allWaterQaSamples
    .filter(ee.Filter.eq('qa_suspect', 1))
    .style({color: 'FF0000', pointSize: 6}),
  {},
  '人工质检-优先核查水体点',
  true
);
Map.addLayer(
  waterSampleTypeRaster.selfMask(),
  {
    min: 1,
    max: 3,
    palette: ['00FFFF', '0055FF', 'FF00FF']
  },
  '水体样本类型栅格：训练/调参/测试',
  false
);

// ============================================================================
// 8. 第二层：非水体四分类调参
// ============================================================================

var nonwaterFeatureNames = nonwaterPredictors.bandNames();
var nonwaterCandidates = [
  {id: 1, mtry: 10, minLeaf: 1},
  {id: 2, mtry: 10, minLeaf: 3},
  {id: 3, mtry: 10, minLeaf: 5},
  {id: 4, mtry: 35, minLeaf: 1},
  {id: 5, mtry: 35, minLeaf: 3},
  {id: 6, mtry: 35, minLeaf: 5}
];

var nonwaterTuneRows = nonwaterCandidates.map(function(config) {
  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: N_TREES,
    variablesPerSplit: config.mtry,
    minLeafPopulation: config.minLeaf,
    bagFraction: BAG_FRACTION,
    seed: RANDOM_SEED
  }).train({
    features: nonwaterTrain,
    classProperty: 'nonwater_label',
    inputProperties: nonwaterFeatureNames
  });

  var assessed = nonwaterTune.classify(classifier, 'nonwater_prediction');
  var metrics = multiclassMetrics(
    assessed,
    'nonwater_label',
    'nonwater_prediction',
    [1, 2, 3, 4]
  );
  var selectionScore = ee.Number(metrics.get('macro_f1')).multiply(1e6)
    .add(ee.Number(metrics.get('overall_accuracy')).multiply(1e3))
    .add(ee.Number(metrics.get('kappa')));

  return ee.Feature(null, {
    record_type: 'nonwater_tuning',
    candidate_id: config.id,
    variablesPerSplit: config.mtry,
    minLeafPopulation: config.minLeaf,
    macro_f1: metrics.get('macro_f1'),
    macro_recall: metrics.get('macro_recall'),
    overall_accuracy: metrics.get('overall_accuracy'),
    kappa: metrics.get('kappa'),
    selection_score: selectionScore
  });
});

var nonwaterTuningTable = ee.FeatureCollection(nonwaterTuneRows)
  .sort('selection_score', false);
var bestNonwater = ee.Feature(nonwaterTuningTable.first());
print('非水体模型调参结果：', nonwaterTuningTable);
print('最优非水体模型：', bestNonwater);

var finalNonwaterTrain = nonwaterTrain.merge(nonwaterTune);
var finalNonwaterClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: N_TREES,
  variablesPerSplit: ee.Number(bestNonwater.get('variablesPerSplit')),
  minLeafPopulation: ee.Number(bestNonwater.get('minLeafPopulation')),
  bagFraction: BAG_FRACTION,
  seed: RANDOM_SEED
}).train({
  features: finalNonwaterTrain,
  classProperty: 'nonwater_label',
  inputProperties: nonwaterFeatureNames
});

var nonwaterPrediction = nonwaterPredictors
  .classify(finalNonwaterClassifier)
  .rename('nonwater_prediction');
var hierarchicalModelClass = nonwaterPrediction
  .where(finalWaterMask, WATER_MODEL_CODE)
  .rename('classification')
  .toInt16();
var finalOriginalClass = hierarchicalModelClass
  .remap(MODEL_CLASSES, ORIGINAL_CLASSES)
  .rename('landcover')
  .toInt16();

// ============================================================================
// 9. 独立测试
// ============================================================================

// 五分类水体测试点使用人工确认的独立水体样本；
// 林地、耕地、建筑和其他仍使用原空间独立测试点。
var reviewedConfirmedWaterTest = reviewedWaterTest
  .filter(ee.Filter.eq('water_label', 1))
  .map(function(feature) {
    return feature.set('model_label', WATER_MODEL_CODE);
  });
var nonwaterClassTestPoints = rawTestPoints.filter(
  ee.Filter.neq('model_label', WATER_MODEL_CODE)
);
var combinedClassTestPoints = nonwaterClassTestPoints.merge(
  reviewedConfirmedWaterTest
);

var testAssessed = hierarchicalModelClass
  .addBands(finalWaterProbability)
  .sampleRegions({
    collection: combinedClassTestPoints,
    properties: ['model_label', 'grid_id'],
    scale: SCALE,
    projection: sampleProjection,
    tileScale: 4,
    geometries: true
  });

var finalMetrics = multiclassMetrics(
  testAssessed,
  'model_label',
  'classification',
  MODEL_CLASSES
);
var finalF1 = ee.List(finalMetrics.get('f1'));

// 人工修正水体测试集的独立二分类评价。
var reviewedWaterTestAssessed = finalWaterProbability.sampleRegions({
  collection: reviewedWaterTest,
  properties: [
    'water_label', 'samp_type', 'grid_id',
    'qa_note', 'reviewer', 'rev_date'
  ],
  scale: SCALE,
  projection: sampleProjection,
  tileScale: 4,
  geometries: true
}).map(function(feature) {
  return feature.set(
    'water_prediction',
    ee.Number(feature.get('water_probability')).gte(bestWaterThreshold).int()
  );
});
var reviewedWaterTestMetrics = multiclassMetrics(
  reviewedWaterTestAssessed,
  'water_label',
  'water_prediction',
  [0, 1]
);
var reviewedWaterTestF1 = ee.List(reviewedWaterTestMetrics.get('f1'));
var reviewedWaterTestProducer = ee.List(
  reviewedWaterTestMetrics.get('producer_accuracy')
);
var reviewedWaterTestUser = ee.List(
  reviewedWaterTestMetrics.get('user_accuracy')
);

var testCropland = testAssessed.filter(
  ee.Filter.eq('model_label', CROPLAND_MODEL_CODE)
);
var croplandPredictedWater = testCropland.filter(
  ee.Filter.eq('classification', WATER_MODEL_CODE)
).size();
var croplandToWaterRate = ee.Number(ee.Algorithms.If(
  testCropland.size().gt(0),
  ee.Number(croplandPredictedWater).divide(testCropland.size()),
  0
));

var testWater = testAssessed.filter(
  ee.Filter.eq('model_label', WATER_MODEL_CODE)
);
var waterPredictedCropland = testWater.filter(
  ee.Filter.eq('classification', CROPLAND_MODEL_CODE)
).size();
var waterToCroplandRate = ee.Number(ee.Algorithms.If(
  testWater.size().gt(0),
  ee.Number(waterPredictedCropland).divide(testWater.size()),
  0
));

print('增强版独立测试混淆矩阵：', finalMetrics.get('matrix'));
print('增强版 OA：', finalMetrics.get('overall_accuracy'));
print('增强版 Kappa：', finalMetrics.get('kappa'));
print('增强版宏平均 F1：', finalMetrics.get('macro_f1'));
print('水体 F1：', finalF1.get(0));
print('耕地 F1：', finalF1.get(2));
print('耕地误判为水体比例：', croplandToWaterRate);
print('水体误判为耕地比例：', waterToCroplandRate);
print(
  '人工水体测试集二分类混淆矩阵：',
  reviewedWaterTestMetrics.get('matrix')
);
print(
  '人工水体测试集二分类 OA：',
  reviewedWaterTestMetrics.get('overall_accuracy')
);
print('人工水体测试集水体 F1：', reviewedWaterTestF1.get(1));
print(
  '人工水体测试集非水体误判水体率：',
  ee.Number(1).subtract(ee.Number(reviewedWaterTestProducer.get(0)))
);
print(
  '人工水体测试集水体漏判率：',
  ee.Number(1).subtract(ee.Number(reviewedWaterTestProducer.get(1)))
);

// ============================================================================
// 10. 置信度、面积和显示
// ============================================================================

var nonwaterProbabilityArray = nonwaterPredictors.classify(
  finalNonwaterClassifier.setOutputMode('MULTIPROBABILITY')
);
var nonwaterMaxProbability = nonwaterProbabilityArray
  .arrayReduce(ee.Reducer.max(), [0])
  .arrayGet([0]);
var confidence = nonwaterMaxProbability
  .multiply(ee.Image(1).subtract(finalWaterProbability))
  .where(finalWaterMask, finalWaterProbability)
  .rename('confidence');

var cropland = finalOriginalClass.eq(5).rename('cropland').toByte();

Map.addLayer(
  hierarchicalModelClass,
  {min: 0, max: 4, palette: CLASS_PALETTE},
  '增强版分层RF分类',
  true
);
Map.addLayer(
  finalWaterProbability,
  {min: 0, max: 1, palette: ['FFFFFF', '80B1D3', '08519C']},
  '永久水体概率',
  false
);
Map.addLayer(
  finalWaterPrediction.selfMask(),
  {min: 1, max: 1, palette: ['0066FF']},
  '最终二值水体预测',
  false
);
Map.addLayer(
  cropland.selfMask(),
  {palette: ['FFD700']},
  '增强版耕地',
  true
);
Map.addLayer(
  confidence,
  {min: 0.2, max: 1, palette: ['D73027', 'FEE08B', '1A9850']},
  '增强版分类置信度',
  false
);

function areaFeature(modelCode, originalCode, name) {
  var predicted = finalOriginalClass.eq(originalCode)
    .multiply(ee.Image.pixelArea())
    .rename('area')
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: region,
      scale: SCALE,
      crs: EXPORT_CRS,
      maxPixels: 1e11,
      tileScale: 4
    });
  var referenceArea = rawLabel.eq(originalCode)
    .multiply(ee.Image.pixelArea())
    .rename('area')
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: region,
      scale: SCALE,
      crs: EXPORT_CRS,
      maxPixels: 1e11,
      tileScale: 4
    });
  var predictedM2 = ee.Number(predicted.get('area'));
  var referenceM2 = ee.Number(referenceArea.get('area'));
  var difference = predictedM2.subtract(referenceM2);
  return ee.Feature(null, {
    model_code: modelCode,
    original_code: originalCode,
    class_name: name,
    predicted_area_m2: predictedM2,
    predicted_area_ha: predictedM2.divide(1e4),
    predicted_area_km2: predictedM2.divide(1e6),
    reference_area_m2: referenceM2,
    reference_area_ha: referenceM2.divide(1e4),
    difference_m2: difference,
    difference_ha: difference.divide(1e4),
    change_percent: ee.Algorithms.If(
      referenceM2.gt(0),
      difference.divide(referenceM2).multiply(100),
      null
    )
  });
}

var areaRows = [];
for (var areaIndex = 0; areaIndex < MODEL_CLASSES.length; areaIndex++) {
  areaRows.push(areaFeature(
    MODEL_CLASSES[areaIndex],
    ORIGINAL_CLASSES[areaIndex],
    CLASS_NAMES[areaIndex]
  ));
}
var areaTable = ee.FeatureCollection(areaRows);
print('增强版各类别面积：', areaTable);

// ============================================================================
// 11. 精度、混淆矩阵和变量重要性表
// ============================================================================

var producer = ee.List(finalMetrics.get('producer_accuracy'));
var user = ee.List(finalMetrics.get('user_accuracy'));
var f1 = ee.List(finalMetrics.get('f1'));

var summaryFeature = ee.Feature(null, {
  record_type: 'final_summary',
  overall_accuracy: finalMetrics.get('overall_accuracy'),
  kappa: finalMetrics.get('kappa'),
  macro_f1: finalMetrics.get('macro_f1'),
  macro_recall: finalMetrics.get('macro_recall'),
  water_f1: f1.get(0),
  cropland_f1: f1.get(2),
  cropland_to_water_rate: croplandToWaterRate,
  water_to_cropland_rate: waterToCroplandRate,
  water_probability_threshold: bestWaterThreshold,
  reviewed_water_binary_oa: reviewedWaterTestMetrics.get('overall_accuracy'),
  reviewed_water_binary_kappa: reviewedWaterTestMetrics.get('kappa'),
  reviewed_water_f1: reviewedWaterTestF1.get(1),
  reviewed_nonwater_f1: reviewedWaterTestF1.get(0),
  reviewed_water_precision: reviewedWaterTestUser.get(1),
  reviewed_water_recall: reviewedWaterTestProducer.get(1),
  reviewed_nonwater_false_positive_rate: ee.Number(1).subtract(
    ee.Number(reviewedWaterTestProducer.get(0))
  ),
  reviewed_water_false_negative_rate: ee.Number(1).subtract(
    ee.Number(reviewedWaterTestProducer.get(1))
  )
});

var reviewedWaterBinarySummary = ee.Feature(null, {
  record_type: 'reviewed_water_binary_test',
  test_sample_count: reviewedWaterTestAssessed.size(),
  overall_accuracy: reviewedWaterTestMetrics.get('overall_accuracy'),
  kappa: reviewedWaterTestMetrics.get('kappa'),
  nonwater_f1: reviewedWaterTestF1.get(0),
  water_f1: reviewedWaterTestF1.get(1),
  water_precision: reviewedWaterTestUser.get(1),
  water_recall: reviewedWaterTestProducer.get(1),
  nonwater_false_positive_rate: ee.Number(1).subtract(
    ee.Number(reviewedWaterTestProducer.get(0))
  ),
  water_false_negative_rate: ee.Number(1).subtract(
    ee.Number(reviewedWaterTestProducer.get(1))
  ),
  water_probability_threshold: bestWaterThreshold
});

var classMetricRows = MODEL_CLASSES.map(function(code, index) {
  return ee.Feature(null, {
    record_type: 'class_accuracy',
    model_code: code,
    original_code: ORIGINAL_CLASSES[index],
    class_name: CLASS_NAMES[index],
    producer_accuracy: producer.get(index),
    user_accuracy: user.get(index),
    f1: f1.get(index),
    test_sample_count: testAssessed
      .filter(ee.Filter.eq('model_label', code))
      .size()
  });
});

var matrixList = ee.List(finalMetrics.get('matrix'));
var matrixRows = [];
for (var actualIndex = 0; actualIndex < MODEL_CLASSES.length; actualIndex++) {
  for (var predictedIndex = 0; predictedIndex < MODEL_CLASSES.length; predictedIndex++) {
    matrixRows.push(ee.Feature(null, {
      record_type: 'confusion_matrix',
      actual_model_code: MODEL_CLASSES[actualIndex],
      actual_name: CLASS_NAMES[actualIndex],
      predicted_model_code: MODEL_CLASSES[predictedIndex],
      predicted_name: CLASS_NAMES[predictedIndex],
      sample_count: ee.List(matrixList.get(actualIndex)).get(predictedIndex)
    }));
  }
}

var accuracyTable = ee.FeatureCollection([summaryFeature])
  .merge(ee.FeatureCollection([reviewedWaterBinarySummary]))
  .merge(ee.FeatureCollection(classMetricRows))
  .merge(ee.FeatureCollection(matrixRows))
  .merge(waterTuningTable)
  .merge(nonwaterTuningTable);

function importanceFeatures(classifier, stage) {
  var importance = ee.Dictionary(
    ee.Dictionary(classifier.explain()).get('importance')
  );
  return ee.FeatureCollection(importance.keys().map(function(name) {
    return ee.Feature(null, {
      model_stage: stage,
      feature: name,
      importance: importance.get(name)
    });
  }));
}

var importanceTable = importanceFeatures(finalWaterClassifier, 'water_binary')
  .merge(importanceFeatures(finalNonwaterClassifier, 'nonwater_four_class'));

// ============================================================================
// 12. 导出
// ============================================================================

Export.image.toDrive({
  image: finalOriginalClass.clip(region),
  description: 'Gaoyou_hierarchical_RF_landcover_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_hierarchical_rf_landcover_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Export.image.toDrive({
  image: cropland.clip(region),
  description: 'Gaoyou_hierarchical_RF_cropland_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_hierarchical_rf_cropland_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Export.image.toDrive({
  image: finalWaterProbability.clip(region),
  description: 'Gaoyou_permanent_water_probability_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_permanent_water_probability_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Export.image.toDrive({
  image: finalWaterPrediction.clip(region),
  description: 'Gaoyou_permanent_water_prediction_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_permanent_water_prediction_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Export.image.toDrive({
  image: waterSampleTypeRaster,
  description: 'Gaoyou_water_sample_types_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_water_sample_types_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Export.image.toDrive({
  image: confidence.clip(region),
  description: 'Gaoyou_hierarchical_RF_confidence_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_hierarchical_rf_confidence_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Export.table.toDrive({
  collection: accuracyTable,
  description: 'Gaoyou_hierarchical_RF_accuracy_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_hierarchical_rf_accuracy_2020',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: areaTable,
  description: 'Gaoyou_hierarchical_RF_area_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_hierarchical_rf_area_2020',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: importanceTable,
  description: 'Gaoyou_hierarchical_RF_feature_importance_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_hierarchical_rf_feature_importance_2020',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: rawTestPoints,
  description: 'Gaoyou_hierarchical_RF_independent_test_points_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_hierarchical_rf_independent_test_points_2020',
  fileFormat: 'GeoJSON'
});

Export.table.toDrive({
  collection: reviewedWaterTestAssessed,
  description: 'Gaoyou_reviewed_water_test_predictions_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_reviewed_water_test_predictions_2020',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: waterTrainShp,
  description: 'Gaoyou_water_training_samples_QA_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_water_training_samples_qa_2020',
  fileFormat: 'SHP'
});

Export.table.toDrive({
  collection: waterTuneShp,
  description: 'Gaoyou_water_tuning_samples_QA_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_water_tuning_samples_qa_2020',
  fileFormat: 'SHP'
});

Export.table.toDrive({
  collection: waterTestShp,
  description: 'Gaoyou_water_independent_test_samples_QA_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_water_independent_test_samples_qa_2020',
  fileFormat: 'SHP'
});

Export.table.toDrive({
  collection: allWaterQaSamples,
  description: 'Gaoyou_all_water_samples_QA_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_all_water_samples_qa_2020',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: allWaterShp,
  description: 'Gaoyou_all_water_sample_points_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_all_water_sample_points_2020',
  fileFormat: 'SHP'
});
