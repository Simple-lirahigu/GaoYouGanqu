/**
 * 高邮地区 2020 年随机森林土地利用分类与耕地面积识别
 *
 * 标签源：
 *   projects/ee-yangsimple237/assets/2020tudi，波段 b1
 *
 * 保留类别：
 *   1 水体，2 林地，5 耕地，7 建筑，11 其他
 * 剔除类别：
 *   4、8（不参与训练、调参、验证和参考面积比较）
 *
 * 精度解释：
 *   精度表示随机森林复现 2020tudi 标签的能力，不等同于独立地面真实性精度。
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
var RANDOM_SEED = 20200624;

var CLASS_VALUES = [1, 2, 5, 7, 11];
var CLASS_NAMES = ['水体', '林地', '耕地', '建筑', '其他'];
var CLASS_PALETTE = ['419BDF', '397D49', 'E49635', 'C4281B', 'A59B8F'];
var CROPLAND_CODE = 5;

var TRAIN_POINTS_PER_CLASS = 1000;
var TUNE_POINTS_PER_CLASS = 300;
var TEST_POINTS_PER_CLASS = 300;

var N_TREES = 500;
var BAG_FRACTION = 0.7;

var DRIVE_FOLDER = 'GEE_Gaoyou_Landcover_RF';

var aoi = ee.FeatureCollection(AOI_ASSET);
var region = aoi.geometry();
var referenceRaw = ee.Image(LANDUSE_ASSET).select(LANDUSE_BAND).rename('label');

// 仅保留 1、2、5、7、11。4和8不进入样本及参考面积统计。
var validLabelMask = referenceRaw.eq(1)
  .or(referenceRaw.eq(2))
  .or(referenceRaw.eq(5))
  .or(referenceRaw.eq(7))
  .or(referenceRaw.eq(11));
var reference = referenceRaw.updateMask(validLabelMask).toInt16();

Map.centerObject(aoi, 10);
Map.addLayer(
  aoi.style({color: 'FF0000', fillColor: '00000000'}),
  {},
  '高邮研究区'
);
Map.addLayer(
  reference.remap(CLASS_VALUES, [0, 1, 2, 3, 4]).clip(region),
  {min: 0, max: 4, palette: CLASS_PALETTE},
  '2020tudi 有效标签',
  false
);

print('土地利用波段：', ee.Image(LANDUSE_ASSET).bandNames());
print('保留类别编码：', CLASS_VALUES);
print('剔除类别编码：[4, 8]');
print(
  '有效标签像元数直方图：',
  reference.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: region,
    scale: SCALE,
    crs: EXPORT_CRS,
    maxPixels: 1e11,
    tileScale: 4
  })
);

// ============================================================================
// 1. Sentinel-2 季度光谱与指数特征
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

  return image
    .updateMask(valid)
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

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
  .map(maskS2)
  .map(addS2Indices)
  .select(s2BaseBands);

var s2Annual = s2.median().select(s2BaseBands);

function makeS2Quarter(start, end, prefix) {
  var quarter = s2.filterDate(start, end).median().select(s2BaseBands);
  var filled = quarter.unmask(s2Annual).unmask(0);
  var renamed = s2BaseBands.map(function(name) {
    return prefix + '_' + name;
  });
  return filled.rename(renamed);
}

var s2Features = ee.Image.cat([
  makeS2Quarter('2020-01-01', '2020-04-01', 'S2_Q1'),
  makeS2Quarter('2020-04-01', '2020-07-01', 'S2_Q2'),
  makeS2Quarter('2020-07-01', '2020-10-01', 'S2_Q3'),
  makeS2Quarter('2020-10-01', '2021-01-01', 'S2_Q4')
]);

print('2020年 Sentinel-2 影像数：', s2.size());

// ============================================================================
// 2. Sentinel-1 季度中值与离散度特征
// ============================================================================

function prepareS1(image) {
  var vv = image.select('VV').rename('VV');
  var vh = image.select('VH').rename('VH');
  var difference = vv.subtract(vh).rename('VV_minus_VH');
  return vv.addBands(vh)
    .addBands(difference)
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

var s1BaseBands = ['VV', 'VH', 'VV_minus_VH'];
var s1Reducer = ee.Reducer.median()
  .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true});
var s1ReducedBandNames = [
  'VV_median', 'VH_median', 'VV_minus_VH_median',
  'VV_stdDev', 'VH_stdDev', 'VV_minus_VH_stdDev'
];

// 空集合兜底影像。正常情况下季度影像使用全年统计填补局部空值；
// 如果某季度完全无影像，则整个季度直接使用全年统计。
var s1Fallback = ee.Image.constant([0, 0, 0, 0, 0, 0])
  .rename(s1ReducedBandNames)
  .toFloat();
var s1Annual = ee.Image(ee.Algorithms.If(
  s1.size().gt(0),
  s1.reduce(s1Reducer).select(s1ReducedBandNames),
  s1Fallback
));

function makeS1Quarter(start, end, prefix) {
  var quarterCollection = s1.filterDate(start, end);
  var quarter = ee.Image(ee.Algorithms.If(
    quarterCollection.size().gt(0),
    quarterCollection.reduce(s1Reducer).select(s1ReducedBandNames),
    s1Annual
  ));
  var renamed = s1ReducedBandNames.map(function(name) {
    return prefix + '_' + name;
  });
  return quarter
    .select(s1ReducedBandNames)
    .unmask(s1Annual)
    .unmask(0)
    .rename(renamed);
}

var s1Features = ee.Image.cat([
  makeS1Quarter('2020-01-01', '2020-04-01', 'S1_Q1'),
  makeS1Quarter('2020-04-01', '2020-07-01', 'S1_Q2'),
  makeS1Quarter('2020-07-01', '2020-10-01', 'S1_Q3'),
  makeS1Quarter('2020-10-01', '2021-01-01', 'S1_Q4')
]);

print('2020年 Sentinel-1 影像总数（升轨+降轨）：', s1.size());
print('Sentinel-1 Q1影像数：', s1.filterDate('2020-01-01', '2020-04-01').size());
print('Sentinel-1 Q2影像数：', s1.filterDate('2020-04-01', '2020-07-01').size());
print('Sentinel-1 Q3影像数：', s1.filterDate('2020-07-01', '2020-10-01').size());
print('Sentinel-1 Q4影像数：', s1.filterDate('2020-10-01', '2021-01-01').size());

// 全部预测特征。所有缺失值已由全年合成或0填补，保证可覆盖全研究区。
var predictors = s2Features
  .addBands(s1Features)
  .toFloat()
  .clip(region);
var predictorNames = predictors.bandNames();

print('预测特征数量：', predictorNames.size());
print('预测特征名称：', predictorNames);

// ============================================================================
// 3. 1 km空间网格及60/20/20确定性分区
// ============================================================================

var gridProjection = ee.Projection(EXPORT_CRS).atScale(GRID_SIZE);
var sampleProjection = ee.Projection(EXPORT_CRS).atScale(SCALE);
var gridCoordinates = ee.Image.pixelCoordinates(gridProjection);
var gridX = gridCoordinates.select('x').toInt64();
var gridY = gridCoordinates.select('y').toInt64();

// 使用网格坐标构造稳定哈希。相同网格在每次运行中属于同一分区。
var gridHash = gridX.multiply(73856093)
  .add(gridY.multiply(19349663))
  .add(RANDOM_SEED)
  .abs()
  .mod(100)
  .rename('grid_hash');
var gridId = gridX.multiply(1000000)
  .add(gridY)
  .rename('grid_id')
  .toInt64();

var trainGridMask = gridHash.lt(60);
var tuneGridMask = gridHash.gte(60).and(gridHash.lt(80));
var testGridMask = gridHash.gte(80);

var splitImage = ee.Image(0)
  .where(trainGridMask, 1)
  .where(tuneGridMask, 2)
  .where(testGridMask, 3)
  .rename('spatial_split')
  .clip(region);

Map.addLayer(
  splitImage,
  {min: 1, max: 3, palette: ['4DAF4A', '377EB8', 'E41A1C']},
  '1 km空间分区：训练/调参/测试',
  false
);

// ============================================================================
// 4. 分层随机抽样
// ============================================================================

var sampleSource = predictors
  .addBands(reference)
  .addBands(gridId);

function stratifiedSampleBySplit(splitMask, pointsPerClass, splitName, seedOffset) {
  var requested = CLASS_VALUES.map(function() {
    return pointsPerClass;
  });

  return sampleSource
    .updateMask(splitMask)
    .stratifiedSample({
      numPoints: 0,
      classBand: 'label',
      classValues: CLASS_VALUES,
      classPoints: requested,
      region: region,
      scale: SCALE,
      projection: sampleProjection,
      seed: RANDOM_SEED + seedOffset,
      dropNulls: true,
      tileScale: 4,
      geometries: true
    })
    .map(function(feature) {
      return feature.set('sample_split', splitName);
    });
}

var trainingSamples = stratifiedSampleBySplit(
  trainGridMask,
  TRAIN_POINTS_PER_CLASS,
  'train',
  1
);
var tuningSamples = stratifiedSampleBySplit(
  tuneGridMask,
  TUNE_POINTS_PER_CLASS,
  'tune',
  2
);
var testingSamples = stratifiedSampleBySplit(
  testGridMask,
  TEST_POINTS_PER_CLASS,
  'test',
  3
);

function sampleDiagnostics(samples, splitName) {
  return ee.FeatureCollection(CLASS_VALUES.map(function(code, index) {
    var oneClass = samples.filter(ee.Filter.eq('label', code));
    return ee.Feature(null, {
      record_type: 'sample_diagnostic',
      sample_split: splitName,
      class_code: code,
      class_name: CLASS_NAMES[index],
      sample_count: oneClass.size(),
      distinct_grid_count: oneClass.aggregate_count_distinct('grid_id')
    });
  }));
}

var sampleDiagnosticsTable = sampleDiagnostics(trainingSamples, 'train')
  .merge(sampleDiagnostics(tuningSamples, 'tune'))
  .merge(sampleDiagnostics(testingSamples, 'test'));

print('样本诊断：', sampleDiagnosticsTable);
print('训练样本数：', trainingSamples.size());
print('调参样本数：', tuningSamples.size());
print('测试样本数：', testingSamples.size());

Map.addLayer(trainingSamples.style({color: '00A600', pointSize: 2}), {}, '训练样本', false);
Map.addLayer(tuningSamples.style({color: '0066FF', pointSize: 2}), {}, '调参样本', false);
Map.addLayer(testingSamples.style({color: 'FF0000', pointSize: 2}), {}, '独立测试样本', false);

// ============================================================================
// 5. 精度指标函数
// ============================================================================

function listMean(values) {
  return ee.Number(ee.List(values).reduce(ee.Reducer.mean()));
}

function metricsFromClassifiedSamples(classifiedSamples) {
  var matrix = classifiedSamples.errorMatrix('label', 'classification', CLASS_VALUES);
  var producerList = ee.Array(matrix.producersAccuracy()).toList().flatten();
  var userList = ee.Array(matrix.consumersAccuracy()).toList().flatten();

  var f1List = ee.List.sequence(0, CLASS_VALUES.length - 1).map(function(i) {
    i = ee.Number(i);
    var producer = ee.Number(producerList.get(i));
    var user = ee.Number(userList.get(i));
    var denominator = producer.add(user);
    return ee.Number(ee.Algorithms.If(
      denominator.gt(0),
      producer.multiply(user).multiply(2).divide(denominator),
      0
    ));
  });

  return ee.Dictionary({
    overall_accuracy: matrix.accuracy(),
    kappa: matrix.kappa(),
    macro_f1: listMean(f1List),
    macro_recall: listMean(producerList),
    producer_accuracy: producerList,
    user_accuracy: userList,
    f1: f1List,
    confusion_matrix: matrix.array().toList()
  });
}

// ============================================================================
// 6. 六组RF参数调优
// ============================================================================

// 当前特征为64个S2季度特征+24个S1季度特征，共88个。
// sqrt(88)约为9，88/3约为29。
var MTRY_SQRT = 9;
var MTRY_THIRD = 29;
var rfCandidates = [
  {id: 1, variablesPerSplit: MTRY_SQRT, minLeafPopulation: 1},
  {id: 2, variablesPerSplit: MTRY_SQRT, minLeafPopulation: 3},
  {id: 3, variablesPerSplit: MTRY_SQRT, minLeafPopulation: 5},
  {id: 4, variablesPerSplit: MTRY_THIRD, minLeafPopulation: 1},
  {id: 5, variablesPerSplit: MTRY_THIRD, minLeafPopulation: 3},
  {id: 6, variablesPerSplit: MTRY_THIRD, minLeafPopulation: 5}
];

var tuningFeatures = rfCandidates.map(function(config) {
  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: N_TREES,
    variablesPerSplit: config.variablesPerSplit,
    minLeafPopulation: config.minLeafPopulation,
    bagFraction: BAG_FRACTION,
    seed: RANDOM_SEED
  }).train({
    features: trainingSamples,
    classProperty: 'label',
    inputProperties: predictorNames
  });

  var classifiedTune = tuningSamples.classify(classifier);
  var metrics = metricsFromClassifiedSamples(classifiedTune);

  return ee.Feature(null, {
    record_type: 'rf_tuning',
    candidate_id: config.id,
    numberOfTrees: N_TREES,
    variablesPerSplit: config.variablesPerSplit,
    minLeafPopulation: config.minLeafPopulation,
    bagFraction: BAG_FRACTION,
    overall_accuracy: metrics.get('overall_accuracy'),
    kappa: metrics.get('kappa'),
    macro_f1: metrics.get('macro_f1'),
    macro_recall: metrics.get('macro_recall')
  });
});

var tuningResults = ee.FeatureCollection(tuningFeatures);

// 排序键优先级：macro F1 > OA > Kappa。
// 三项均位于[0,1]附近，缩放后组成单一排序分数。
var rankedTuning = tuningResults.map(function(feature) {
  var rankScore = ee.Number(feature.get('macro_f1')).multiply(1e6)
    .add(ee.Number(feature.get('overall_accuracy')).multiply(1e3))
    .add(ee.Number(feature.get('kappa')));
  return feature.set('rank_score', rankScore);
}).sort('rank_score', false);

var bestParameters = ee.Feature(rankedTuning.first());
print('RF调参结果：', rankedTuning);
print('最优RF参数：', bestParameters);

// 将训练集和调参集合并，按最优参数重训最终模型。
var finalTrainingSamples = trainingSamples.merge(tuningSamples);
var finalClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: N_TREES,
  variablesPerSplit: ee.Number(bestParameters.get('variablesPerSplit')),
  minLeafPopulation: ee.Number(bestParameters.get('minLeafPopulation')),
  bagFraction: BAG_FRACTION,
  seed: RANDOM_SEED
}).train({
  features: finalTrainingSamples,
  classProperty: 'label',
  inputProperties: predictorNames
});

// 独立测试集只在最终模型确定后评价一次。
var classifiedTest = testingSamples.classify(finalClassifier);
var finalMetrics = metricsFromClassifiedSamples(classifiedTest);
var finalMatrix = classifiedTest.errorMatrix('label', 'classification', CLASS_VALUES);

print('独立测试混淆矩阵：', finalMatrix);
print('独立测试总体精度 OA：', finalMetrics.get('overall_accuracy'));
print('独立测试 Kappa：', finalMetrics.get('kappa'));
print('独立测试宏平均 F1：', finalMetrics.get('macro_f1'));
print('独立测试宏平均召回率：', finalMetrics.get('macro_recall'));
print('各类生产者精度：', finalMetrics.get('producer_accuracy'));
print('各类用户精度：', finalMetrics.get('user_accuracy'));
print('各类 F1：', finalMetrics.get('f1'));

// ============================================================================
// 7. 精度表与变量重要性
// ============================================================================

var finalSummary = ee.Feature(null, {
  record_type: 'final_summary',
  class_code: -1,
  class_name: '全部类别',
  overall_accuracy: finalMetrics.get('overall_accuracy'),
  kappa: finalMetrics.get('kappa'),
  macro_f1: finalMetrics.get('macro_f1'),
  macro_recall: finalMetrics.get('macro_recall'),
  numberOfTrees: N_TREES,
  variablesPerSplit: bestParameters.get('variablesPerSplit'),
  minLeafPopulation: bestParameters.get('minLeafPopulation'),
  bagFraction: BAG_FRACTION
});

var producerAccuracy = ee.List(finalMetrics.get('producer_accuracy'));
var userAccuracy = ee.List(finalMetrics.get('user_accuracy'));
var f1Accuracy = ee.List(finalMetrics.get('f1'));

var classAccuracyFeatures = ee.FeatureCollection(CLASS_VALUES.map(function(code, index) {
  var testClassCount = testingSamples.filter(ee.Filter.eq('label', code)).size();
  var gridCount = testingSamples
    .filter(ee.Filter.eq('label', code))
    .aggregate_count_distinct('grid_id');
  var estimable = ee.Number(testClassCount).multiply(ee.Number(gridCount)).gt(0);

  return ee.Feature(null, {
    record_type: 'class_accuracy',
    class_code: code,
    class_name: CLASS_NAMES[index],
    producer_accuracy: ee.Algorithms.If(estimable, producerAccuracy.get(index), null),
    user_accuracy: ee.Algorithms.If(estimable, userAccuracy.get(index), null),
    f1: ee.Algorithms.If(estimable, f1Accuracy.get(index), null),
    test_sample_count: testClassCount,
    test_grid_count: gridCount,
    spatial_accuracy_estimable: estimable
  });
}));

var confusionArray = finalMatrix.array();
var confusionMatrixFeatures = ee.FeatureCollection(
  ee.List.sequence(0, CLASS_VALUES.length - 1).map(function(row) {
    row = ee.Number(row);
    return ee.List.sequence(0, CLASS_VALUES.length - 1).map(function(column) {
      column = ee.Number(column);
      return ee.Feature(null, {
        record_type: 'confusion_matrix',
        actual_class: ee.List(CLASS_VALUES).get(row),
        predicted_class: ee.List(CLASS_VALUES).get(column),
        sample_count: confusionArray.get([row, column])
      });
    });
  }).flatten()
);

var accuracyTable = ee.FeatureCollection([finalSummary])
  .merge(classAccuracyFeatures)
  .merge(confusionMatrixFeatures)
  .merge(rankedTuning)
  .merge(sampleDiagnosticsTable);

var explanation = ee.Dictionary(finalClassifier.explain());
var importance = ee.Dictionary(explanation.get('importance'));
var importanceTable = ee.FeatureCollection(importance.keys().map(function(name) {
  return ee.Feature(null, {
    feature: name,
    importance: importance.get(name)
  });
})).sort('importance', false);

print('变量重要性：', importanceTable);

// ============================================================================
// 8. 全高邮分类、置信度和耕地提取
// ============================================================================

var classified = predictors
  .classify(finalClassifier)
  .rename('landcover')
  .toInt16();
var classifiedDisplay = classified.remap(CLASS_VALUES, [0, 1, 2, 3, 4]);

var probabilityClassifier = finalClassifier.setOutputMode('MULTIPROBABILITY');
var probabilityArray = predictors.classify(probabilityClassifier);
var confidence = probabilityArray
  .arrayReduce(ee.Reducer.max(), [0])
  .arrayGet([0])
  .rename('confidence');

var cropland = classified.eq(CROPLAND_CODE)
  .rename('cropland')
  .toByte();

Map.addLayer(
  classifiedDisplay,
  {min: 0, max: 4, palette: CLASS_PALETTE},
  'RF土地利用分类 2020',
  true
);
Map.addLayer(
  cropland.selfMask(),
  {palette: ['FFD700']},
  'RF耕地',
  true
);
Map.addLayer(
  confidence,
  {min: 0.2, max: 1, palette: ['D73027', 'FEE08B', '1A9850']},
  '分类置信度',
  false
);

// ============================================================================
// 9. 分类面积、参考面积及差值
// ============================================================================

function classAreaFeature(code, name) {
  var predictedArea = classified.eq(code)
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

  var referenceArea = reference.eq(code)
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

  var predictedM2 = ee.Number(predictedArea.get('area'));
  var referenceM2 = ee.Number(referenceArea.get('area'));
  var differenceM2 = predictedM2.subtract(referenceM2);
  var changePercent = ee.Algorithms.If(
    referenceM2.gt(0),
    differenceM2.divide(referenceM2).multiply(100),
    null
  );

  return ee.Feature(null, {
    class_code: code,
    class_name: name,
    predicted_area_m2: predictedM2,
    predicted_area_ha: predictedM2.divide(1e4),
    predicted_area_km2: predictedM2.divide(1e6),
    reference_area_m2: referenceM2,
    reference_area_ha: referenceM2.divide(1e4),
    reference_area_km2: referenceM2.divide(1e6),
    difference_m2: differenceM2,
    difference_ha: differenceM2.divide(1e4),
    difference_km2: differenceM2.divide(1e6),
    change_percent: changePercent
  });
}

var areaTable = ee.FeatureCollection(CLASS_VALUES.map(function(code, index) {
  return classAreaFeature(code, CLASS_NAMES[index]);
}));

var predictedTotalM2 = ee.Number(areaTable.aggregate_sum('predicted_area_m2'));
var validReferenceTotalM2 = ee.Number(areaTable.aggregate_sum('reference_area_m2'));
var aoiAreaM2 = region.area(1);

print('各类别面积统计：', areaTable);
print('预测分类面积总和（m²）：', predictedTotalM2);
print('有效参考类别面积总和（m²）：', validReferenceTotalM2);
print('AOI几何面积（m²）：', aoiAreaM2);
print(
  '耕地面积（ha）：',
  ee.Feature(areaTable.filter(ee.Filter.eq('class_code', CROPLAND_CODE)).first())
    .get('predicted_area_ha')
);

// ============================================================================
// 10. 导出
// ============================================================================

Export.image.toDrive({
  image: classified.clip(region),
  description: 'Gaoyou_landcover_RF_2020_10m',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_landcover_rf_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Export.image.toDrive({
  image: cropland.clip(region),
  description: 'Gaoyou_cropland_RF_2020_10m',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_cropland_rf_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Export.image.toDrive({
  image: confidence.clip(region),
  description: 'Gaoyou_landcover_RF_confidence_2020_10m',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_landcover_rf_confidence_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Export.table.toDrive({
  collection: areaTable,
  description: 'Gaoyou_landcover_area_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_landcover_area_2020',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: accuracyTable,
  description: 'Gaoyou_landcover_accuracy_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_landcover_accuracy_2020',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: importanceTable,
  description: 'Gaoyou_RF_feature_importance_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_rf_feature_importance_2020',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: trainingSamples,
  description: 'Gaoyou_RF_training_samples_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_rf_training_samples_2020',
  fileFormat: 'GeoJSON'
});

Export.table.toDrive({
  collection: tuningSamples,
  description: 'Gaoyou_RF_tuning_samples_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_rf_tuning_samples_2020',
  fileFormat: 'GeoJSON'
});

Export.table.toDrive({
  collection: testingSamples,
  description: 'Gaoyou_RF_testing_samples_2020',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'gaoyou_rf_testing_samples_2020',
  fileFormat: 'GeoJSON'
});
