/**
 * 高邮地区耕地/非耕地增强二分类脚本
 *
 * 目标：
 * - 将原五类土地利用问题简化为“耕地 vs 非耕地”二分类；
 * - 融合 Sentinel-2 光学指数、Sentinel-1 SAR 时序和物候特征；
 * - 当前默认只用 RF 跑通流程；GTB 代码保留，后续需要时再开启对比；
 * - 输出 OA、Kappa、PA、UA、F1、Macro-F1 和耕地面积；
 * - 默认一年一个模型运行，避免 GEE 容量超限。
 *
 * 重要说明：
 * - 当前精度表示模型复现土地利用资产标签的能力，不等同于独立地面真实性精度；
 * - 若要发表高水平论文，建议后续增加人工目视验证样本；
 * - 2021—2025 的土地利用资产路径需要先在参数区填写。
 */

// ============================================================================
// 0. 每次运行通常只需要修改这里
// ============================================================================

// 目标年份。逐年运行时只改这一行，例如 2021、2022、2023。
var TARGET_YEAR = 2020;

// 模型名称：先只用 'RF' 跑通流程；后续如需对比增强模型，可改为 'GTB'。
var MODEL_NAME = 'RF';

// 默认制图阈值。P(cropland) >= 该阈值时判为耕地。
var CLASSIFICATION_THRESHOLD = 0.50;

// 阈值优化列表。一次运行会同时输出这些阈值下的 OA、Kappa、PA、UA、F1。
var THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65];

// 是否额外导出当前年份的耕地二值 GeoTIFF。只做精度表时保持 false。
var EXPORT_CROPLAND_TIF = false;

// 是否尝试一次性生成多年预测耕地面积表。
// 默认 false。若改为 true，会对 YEAR_LIST 中每一年重新训练并分类，可能较慢。
var EXPORT_MULTI_YEAR_AREA_TABLE = false;

// ============================================================================
// 1. 数据资产与基础参数
// ============================================================================

var AOI_ASSET = 'projects/ee-yangsimple237/assets/GYBJ';
var LANDUSE_ASSETS_BY_YEAR = {
  '2020': 'projects/ee-yangsimple237/assets/2020tudi',
  '2021': '',
  '2022': '',
  '2023': '',
  '2024': '',
  '2025': ''
};

var YEAR_LIST = [2020, 2021, 2022, 2023, 2024, 2025];
var LANDUSE_BAND = 'b1';

// 原始土地利用编码：5 为耕地；1、2、7、11 合并为非耕地；4、8 剔除。
var CROPLAND_CODE = 5;
var VALID_ORIGINAL_CLASSES = [1, 2, 5, 7, 11];

// 二分类编码：0=非耕地，1=耕地。
var CLASS_VALUES = [0, 1];
var CLASS_NAMES = ['non_cropland', 'cropland'];
var CLASS_PALETTE = ['BDBDBD', 'E49635'];

var SCALE = 10;
var GRID_SIZE = 1000;
var EXPORT_CRS = 'EPSG:32650';
var RANDOM_SEED = 20200625;
var DRIVE_FOLDER = 'GEE_Gaoyou_Cropland_Binary';

// 样本数。二分类比五分类更轻，可以适当增加；若 GEE 报容量错误，可降到 1500。
var RANDOM_POINTS_PER_CLASS = 2500;
var RANDOM_TRAIN_RATIO = 0.70;
var SPATIAL_TRAIN_PER_CLASS = 2500;
var SPATIAL_TEST_PER_CLASS = 1000;

// RF 参数。
var RF_TREES = 700;
var RF_MTRY = 12;
var RF_MIN_LEAF = 2;
var RF_BAG_FRACTION = 0.7;

// GTB 参数暂时保留，当前默认不启用；后续需要模型对比时再将 MODEL_NAME 改为 'GTB'。
var GTB_TREES = 300;
var GTB_SHRINKAGE = 0.05;
var GTB_SAMPLING_RATE = 0.7;
var GTB_MAX_NODES = 64;

var aoi = ee.FeatureCollection(AOI_ASSET);
var region = aoi.geometry();
var targetYearText = String(TARGET_YEAR);
var targetLanduseAsset = LANDUSE_ASSETS_BY_YEAR[targetYearText];

if (!targetLanduseAsset) {
  throw new Error('请先在 LANDUSE_ASSETS_BY_YEAR 中填写 TARGET_YEAR 对应土地利用资产。');
}

Map.centerObject(aoi, 10);
Map.addLayer(
  aoi.style({color: 'FF0000', fillColor: '00000000'}),
  {},
  'Gaoyou AOI'
);

// ============================================================================
// 2. 标签构建：耕地/非耕地二分类
// ============================================================================

function makeBinaryLabel(year) {
  var yearText = String(year);
  var asset = LANDUSE_ASSETS_BY_YEAR[yearText];
  if (!asset) {
    throw new Error('YEAR_LIST 中的 ' + yearText + ' 尚未配置土地利用资产。');
  }

  var raw = ee.Image(asset).select(LANDUSE_BAND);
  var validMask = raw.eq(1)
    .or(raw.eq(2))
    .or(raw.eq(5))
    .or(raw.eq(7))
    .or(raw.eq(11));

  return raw.eq(CROPLAND_CODE)
    .rename('label')
    .updateMask(validMask)
    .toInt16();
}

// ============================================================================
// 3. Sentinel-2 光学指数与物候特征
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
  var lswi = image.normalizedDifference(['nir', 'swir1']).rename('LSWI');
  var mndwi = image.normalizedDifference(['green', 'swir1']).rename('MNDWI');
  var ndbi = image.normalizedDifference(['swir1', 'nir']).rename('NDBI');
  var ndre = image.normalizedDifference(['nirNarrow', 're1']).rename('NDRE');

  return image.addBands([ndvi, evi, lswi, mndwi, ndbi, ndre]);
}

function makeS2Collection(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');

  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
    .map(maskS2)
    .map(addS2Indices);
}

function safeS2Composite(collection, start, end, bands, fallback) {
  var subset = collection.filterDate(start, end);
  return ee.Image(ee.Algorithms.If(
    subset.size().gt(0),
    subset.median().select(bands),
    fallback.select(bands)
  )).unmask(fallback.select(bands)).unmask(0);
}

function renameBands(image, prefix) {
  return image.rename(image.bandNames().map(function(name) {
    return ee.String(prefix).cat('_').cat(ee.String(name));
  }));
}

function makeS2Features(year) {
  var yearText = String(year);
  var s2 = makeS2Collection(year);
  var opticalBands = [
    'blue', 'green', 'red', 're1', 're2', 're3',
    'nir', 'nirNarrow', 'swir1', 'swir2',
    'NDVI', 'EVI', 'LSWI', 'MNDWI', 'NDBI', 'NDRE'
  ];
  var indexBands = ['NDVI', 'EVI', 'LSWI', 'MNDWI', 'NDBI', 'NDRE'];
  var annual = s2.median().select(opticalBands).unmask(0);

  var quarters = ee.Image.cat([
    renameBands(
      safeS2Composite(s2, yearText + '-01-01', yearText + '-04-01', opticalBands, annual),
      'Q1'
    ),
    renameBands(
      safeS2Composite(s2, yearText + '-04-01', yearText + '-07-01', opticalBands, annual),
      'Q2'
    ),
    renameBands(
      safeS2Composite(s2, yearText + '-07-01', yearText + '-10-01', opticalBands, annual),
      'Q3'
    ),
    renameBands(
      safeS2Composite(s2, yearText + '-10-01', String(year + 1) + '-01-01', opticalBands, annual),
      'Q4'
    )
  ]);

  var monthlyImages = [];
  for (var month = 1; month <= 12; month++) {
    var start = ee.Date.fromYMD(year, month, 1);
    var end = start.advance(1, 'month');
    var prefix = month < 10 ? 'M0' + month : 'M' + month;
    monthlyImages.push(renameBands(
      safeS2Composite(s2, start, end, indexBands, annual),
      prefix
    ));
  }

  var phenology = s2.select(indexBands).reduce(
    ee.Reducer.mean()
      .combine({reducer2: ee.Reducer.min(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
  ).unmask(0);

  var ndviAmp = phenology.select('NDVI_max')
    .subtract(phenology.select('NDVI_min'))
    .rename('NDVI_amplitude');
  var eviAmp = phenology.select('EVI_max')
    .subtract(phenology.select('EVI_min'))
    .rename('EVI_amplitude');
  var lswiAmp = phenology.select('LSWI_max')
    .subtract(phenology.select('LSWI_min'))
    .rename('LSWI_amplitude');

  return ee.Image.cat([
    quarters,
    ee.Image.cat(monthlyImages),
    renameBands(phenology, 'PHE'),
    ndviAmp,
    eviAmp,
    lswiAmp
  ]).toFloat();
}

// ============================================================================
// 4. Sentinel-1 SAR 时序特征
// ============================================================================

function toNatural(image) {
  return ee.Image(10).pow(image.divide(10));
}

function toDb(image) {
  return image.log10().multiply(10);
}

function preprocessS1(image) {
  var vv = image.select('VV');
  var vh = image.select('VH');
  var vvNat = toNatural(vv);
  var vhNat = toNatural(vh);
  var ratio = vvNat.divide(vhNat).rename('VVVH_ratio');
  var diff = vv.subtract(vh).rename('VV_minus_VH');

  return image.select(['VV', 'VH'])
    .addBands([diff, toDb(ratio).rename('VVVH_ratio_db')])
    .copyProperties(image, ['system:time_start']);
}

function makeS1Collection(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');

  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(region)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH'])
    .map(preprocessS1);
}

function safeS1Composite(collection, start, end, fallback) {
  var bands = ['VV', 'VH', 'VV_minus_VH', 'VVVH_ratio_db'];
  var subset = collection.filterDate(start, end);
  return ee.Image(ee.Algorithms.If(
    subset.size().gt(0),
    subset.median().select(bands),
    fallback.select(bands)
  )).unmask(fallback.select(bands)).unmask(0);
}

function makeS1Features(year) {
  var yearText = String(year);
  var s1 = makeS1Collection(year);
  var s1Bands = ['VV', 'VH', 'VV_minus_VH', 'VVVH_ratio_db'];
  var annual = s1.median().select(s1Bands).unmask(0);

  var quarters = ee.Image.cat([
    renameBands(safeS1Composite(s1, yearText + '-01-01', yearText + '-04-01', annual), 'S1_Q1'),
    renameBands(safeS1Composite(s1, yearText + '-04-01', yearText + '-07-01', annual), 'S1_Q2'),
    renameBands(safeS1Composite(s1, yearText + '-07-01', yearText + '-10-01', annual), 'S1_Q3'),
    renameBands(safeS1Composite(s1, yearText + '-10-01', String(year + 1) + '-01-01', annual), 'S1_Q4')
  ]);

  var monthlyImages = [];
  for (var month = 1; month <= 12; month++) {
    var start = ee.Date.fromYMD(year, month, 1);
    var end = start.advance(1, 'month');
    var prefix = month < 10 ? 'S1_M0' + month : 'S1_M' + month;
    monthlyImages.push(renameBands(
      safeS1Composite(s1, start, end, annual),
      prefix
    ));
  }

  var stats = s1.select(s1Bands).reduce(
    ee.Reducer.mean()
      .combine({reducer2: ee.Reducer.min(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
  ).unmask(0);

  var vhAmp = stats.select('VH_max')
    .subtract(stats.select('VH_min'))
    .rename('S1_VH_amplitude');
  var vvAmp = stats.select('VV_max')
    .subtract(stats.select('VV_min'))
    .rename('S1_VV_amplitude');

  return ee.Image.cat([
    quarters,
    ee.Image.cat(monthlyImages),
    renameBands(stats, 'S1_STAT'),
    vhAmp,
    vvAmp
  ]).toFloat();
}

function makeFeatureImage(year) {
  return makeS2Features(year)
    .addBands(makeS1Features(year))
    .clip(region)
    .toFloat();
}

// ============================================================================
// 5. 样本、模型和精度指标
// ============================================================================

function makeGridMasks() {
  var gridProjection = ee.Projection(EXPORT_CRS).atScale(GRID_SIZE);
  var coordinates = ee.Image.pixelCoordinates(gridProjection);
  var gridX = coordinates.select('x').toInt64();
  var gridY = coordinates.select('y').toInt64();
  var gridId = gridX.multiply(1000000).add(gridY).rename('grid_id').toInt64();
  var gridHash = gridX.multiply(73856093)
    .add(gridY.multiply(19349663))
    .add(RANDOM_SEED)
    .abs()
    .mod(100);

  return {
    gridId: gridId,
    trainMask: gridHash.lt(70),
    testMask: gridHash.gte(70)
  };
}

function buildClassifier(modelName) {
  if (modelName === 'RF') {
    return ee.Classifier.smileRandomForest({
      numberOfTrees: RF_TREES,
      variablesPerSplit: RF_MTRY,
      minLeafPopulation: RF_MIN_LEAF,
      bagFraction: RF_BAG_FRACTION,
      seed: RANDOM_SEED
    });
  }

  if (modelName === 'GTB') {
    return ee.Classifier.smileGradientTreeBoost(
      GTB_TREES,
      GTB_SHRINKAGE,
      GTB_SAMPLING_RATE,
      GTB_MAX_NODES,
      null,
      RANDOM_SEED
    );
  }

  throw new Error('MODEL_NAME 只能为 RF 或 GTB。');
}

function makeSamples(label, gridInfo, validationType, trainOrTest) {
  var sampleProjection = ee.Projection(EXPORT_CRS).atScale(SCALE);
  var labelWithGrid = label.addBands(gridInfo.gridId);

  if (validationType === 'random') {
    var master = labelWithGrid.stratifiedSample({
      numPoints: RANDOM_POINTS_PER_CLASS,
      classBand: 'label',
      region: region,
      scale: SCALE,
      projection: sampleProjection,
      seed: RANDOM_SEED + 1,
      dropNulls: true,
      tileScale: 4,
      geometries: true
    }).randomColumn('random_key', RANDOM_SEED + 2);

    return trainOrTest === 'train'
      ? master.filter(ee.Filter.lt('random_key', RANDOM_TRAIN_RATIO))
      : master.filter(ee.Filter.gte('random_key', RANDOM_TRAIN_RATIO));
  }

  var mask = trainOrTest === 'train' ? gridInfo.trainMask : gridInfo.testMask;
  var n = trainOrTest === 'train' ? SPATIAL_TRAIN_PER_CLASS : SPATIAL_TEST_PER_CLASS;
  var seed = trainOrTest === 'train' ? RANDOM_SEED + 10 : RANDOM_SEED + 11;

  return labelWithGrid.updateMask(mask).stratifiedSample({
    numPoints: n,
    classBand: 'label',
    region: region,
    scale: SCALE,
    projection: sampleProjection,
    seed: seed,
    dropNulls: true,
    tileScale: 4,
    geometries: true
  });
}

function meanList(values) {
  return ee.Number(ee.List(values).reduce(ee.Reducer.mean()));
}

function metricsFromSamples(samples) {
  var matrix = samples.errorMatrix('label', 'classification', CLASS_VALUES);
  var pa = ee.Array(matrix.producersAccuracy()).toList().flatten();
  var ua = ee.Array(matrix.consumersAccuracy()).toList().flatten();
  var f1 = ee.List.sequence(0, CLASS_VALUES.length - 1).map(function(index) {
    index = ee.Number(index);
    var recall = ee.Number(pa.get(index));
    var precision = ee.Number(ua.get(index));
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
    pa: pa,
    ua: ua,
    f1: f1,
    macro_f1: meanList(f1)
  });
}

function addCroplandProbability(feature) {
  var probabilities = ee.Array(feature.get('probability_array'));
  return feature.set('cropland_probability', probabilities.get([1]));
}

function classifyProbabilitySamples(samples, threshold) {
  return samples.map(function(feature) {
    var predicted = ee.Number(feature.get('cropland_probability'))
      .gte(threshold)
      .toInt();
    return feature.set({
      classification: predicted,
      threshold: threshold
    });
  });
}

function thresholdMetricFeature(samplesWithProbability, threshold, year, modelName, validationType) {
  var assessed = classifyProbabilitySamples(samplesWithProbability, threshold);
  var metrics = metricsFromSamples(assessed);
  var pa = ee.List(metrics.get('pa'));
  var ua = ee.List(metrics.get('ua'));
  var f1 = ee.List(metrics.get('f1'));

  return ee.Feature(null, {
    record_type: 'threshold_metric',
    year: year,
    model: modelName,
    validation: validationType,
    threshold: threshold,
    overall_accuracy: metrics.get('overall_accuracy'),
    kappa: metrics.get('kappa'),
    macro_f1: metrics.get('macro_f1'),
    non_cropland_PA: pa.get(0),
    non_cropland_UA: ua.get(0),
    non_cropland_F1: f1.get(0),
    cropland_PA: pa.get(1),
    cropland_UA: ua.get(1),
    cropland_F1: f1.get(1)
  });
}

function areaByClass(classification, year, modelName, validationType) {
  var croplandArea = ee.Image.pixelArea()
    .updateMask(classification.eq(1))
    .rename('area')
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: region,
      scale: SCALE,
      crs: EXPORT_CRS,
      maxPixels: 1e13,
      tileScale: 4
    })
    .getNumber('area');

  var validArea = ee.Image.pixelArea()
    .updateMask(classification.gte(0))
    .rename('area')
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: region,
      scale: SCALE,
      crs: EXPORT_CRS,
      maxPixels: 1e13,
      tileScale: 4
    })
    .getNumber('area');

  croplandArea = ee.Number(ee.Algorithms.If(croplandArea, croplandArea, 0));
  validArea = ee.Number(ee.Algorithms.If(validArea, validArea, 0));

  return ee.Feature(null, {
    record_type: 'area',
    year: year,
    model: modelName,
    validation: validationType,
    threshold: CLASSIFICATION_THRESHOLD,
    cropland_area_m2: croplandArea,
    cropland_area_ha: croplandArea.divide(10000),
    cropland_area_km2: croplandArea.divide(1000000),
    valid_area_m2: validArea,
    cropland_ratio: croplandArea.divide(validArea)
  });
}

function referenceArea(label, year) {
  var croplandArea = ee.Image.pixelArea()
    .updateMask(label.eq(1))
    .rename('area')
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: region,
      scale: SCALE,
      crs: EXPORT_CRS,
      maxPixels: 1e13,
      tileScale: 4
    })
    .getNumber('area');

  var validArea = ee.Image.pixelArea()
    .updateMask(label.gte(0))
    .rename('area')
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: region,
      scale: SCALE,
      crs: EXPORT_CRS,
      maxPixels: 1e13,
      tileScale: 4
    })
    .getNumber('area');

  croplandArea = ee.Number(ee.Algorithms.If(croplandArea, croplandArea, 0));
  validArea = ee.Number(ee.Algorithms.If(validArea, validArea, 0));

  return ee.Feature(null, {
    record_type: 'reference_area',
    year: year,
    model: 'landuse_asset',
    validation: 'reference',
    threshold: -1,
    cropland_area_m2: croplandArea,
    cropland_area_ha: croplandArea.divide(10000),
    cropland_area_km2: croplandArea.divide(1000000),
    valid_area_m2: validArea,
    cropland_ratio: croplandArea.divide(validArea)
  });
}

function evaluateYear(year, modelName, validationType) {
  var label = makeBinaryLabel(year);
  var features = makeFeatureImage(year);
  var gridInfo = makeGridMasks();
  var trainPoints = makeSamples(label, gridInfo, validationType, 'train');
  var testPoints = makeSamples(label, gridInfo, validationType, 'test');
  var featureNames = features.bandNames();
  var sampleProjection = ee.Projection(EXPORT_CRS).atScale(SCALE);

  var trainSamples = features.sampleRegions({
    collection: trainPoints,
    properties: ['label', 'grid_id'],
    scale: SCALE,
    projection: sampleProjection,
    tileScale: 4,
    geometries: false
  });
  var testSamples = features.sampleRegions({
    collection: testPoints,
    properties: ['label', 'grid_id'],
    scale: SCALE,
    projection: sampleProjection,
    tileScale: 4,
    geometries: false
  });

  var classifier = buildClassifier(modelName)
    .setOutputMode('MULTIPROBABILITY')
    .train({
    features: trainSamples,
    classProperty: 'label',
    inputProperties: featureNames
  });

  var samplesWithProbability = testSamples
    .classify(classifier, 'probability_array')
    .map(addCroplandProbability);
  var assessed = classifyProbabilitySamples(
    samplesWithProbability,
    CLASSIFICATION_THRESHOLD
  );
  var metrics = metricsFromSamples(assessed);
  var pa = ee.List(metrics.get('pa'));
  var ua = ee.List(metrics.get('ua'));
  var f1 = ee.List(metrics.get('f1'));
  var matrix = ee.List(metrics.get('matrix'));
  var summary = ee.Feature(null, {
    record_type: 'summary',
    year: year,
    model: modelName,
    validation: validationType,
    feature_count: featureNames.size(),
    train_sample_count: trainSamples.size(),
    test_sample_count: testSamples.size(),
    overall_accuracy: metrics.get('overall_accuracy'),
    kappa: metrics.get('kappa'),
    macro_f1: metrics.get('macro_f1')
  });

  var classRows = [];
  for (var classIndex = 0; classIndex < CLASS_VALUES.length; classIndex++) {
    classRows.push(ee.Feature(null, {
      record_type: 'class_metric',
      year: year,
      model: modelName,
      validation: validationType,
      class_code: CLASS_VALUES[classIndex],
      class_name: CLASS_NAMES[classIndex],
      PA: pa.get(classIndex),
      UA: ua.get(classIndex),
      F1: f1.get(classIndex)
    }));
  }

  var matrixRows = [];
  for (var actualIndex = 0; actualIndex < CLASS_VALUES.length; actualIndex++) {
    for (var predictedIndex = 0; predictedIndex < CLASS_VALUES.length; predictedIndex++) {
      matrixRows.push(ee.Feature(null, {
        record_type: 'confusion_matrix',
        year: year,
        model: modelName,
        validation: validationType,
        actual_code: CLASS_VALUES[actualIndex],
        actual_name: CLASS_NAMES[actualIndex],
        predicted_code: CLASS_VALUES[predictedIndex],
        predicted_name: CLASS_NAMES[predictedIndex],
        sample_count: ee.List(matrix.get(actualIndex)).get(predictedIndex)
      }));
    }
  }

  var thresholdRows = THRESHOLDS.map(function(threshold) {
    return thresholdMetricFeature(
      samplesWithProbability,
      threshold,
      year,
      modelName,
      validationType
    );
  });

  var metricTable = ee.FeatureCollection([summary])
    .merge(ee.FeatureCollection(classRows))
    .merge(ee.FeatureCollection(matrixRows))
    .merge(ee.FeatureCollection(thresholdRows));
  var classification = ee.Image(0).rename('cropland').toInt16();
  var probability = ee.Image(0).rename('cropland_probability').toFloat();
  var areaCollection = ee.FeatureCollection([]);
  var outputTable = metricTable;

  if (validationType === 'spatial') {
    probability = features.classify(classifier)
      .arrayGet([1])
      .rename('cropland_probability')
      .toFloat();
    classification = probability
      .gte(CLASSIFICATION_THRESHOLD)
      .rename('cropland')
      .toInt16();
    areaCollection = ee.FeatureCollection([
      areaByClass(classification, year, modelName, validationType)
    ]);
    outputTable = metricTable.merge(areaCollection);
  }

  return {
    table: outputTable,
    summary: ee.FeatureCollection([summary]),
    classMetric: ee.FeatureCollection(classRows),
    confusion: ee.FeatureCollection(matrixRows),
    thresholdMetric: ee.FeatureCollection(thresholdRows),
    area: areaCollection,
    classification: classification,
    probability: probability,
    label: label,
    trainPoints: trainPoints,
    testPoints: testPoints
  };
}

// ============================================================================
// 6. 当前年份运行、地图显示与导出
// ============================================================================

var randomResult = evaluateYear(TARGET_YEAR, MODEL_NAME, 'random');
var spatialResult = evaluateYear(TARGET_YEAR, MODEL_NAME, 'spatial');
var resultTable = randomResult.table.merge(spatialResult.table);
var summaryTable = randomResult.summary.merge(spatialResult.summary);
var classMetricTable = randomResult.classMetric.merge(spatialResult.classMetric);
var thresholdMetricTable = randomResult.thresholdMetric
  .merge(spatialResult.thresholdMetric);
var prediction = spatialResult.classification;
var probability = spatialResult.probability;
var label = spatialResult.label;
var areaTable = spatialResult.area.merge(ee.FeatureCollection([
  referenceArea(label, TARGET_YEAR)
]));

print('当前年份：', TARGET_YEAR);
print('当前模型：', MODEL_NAME);
print('特征数量：', makeFeatureImage(TARGET_YEAR).bandNames().size());
print('总体精度表：', summaryTable);
print('各类别 PA / UA / F1 表：', classMetricTable);
print('空间验证预测耕地面积：', areaTable);

print('threshold metrics:', thresholdMetricTable);
print(
  'best spatial threshold by cropland F1:',
  thresholdMetricTable
    .filter(ee.Filter.eq('validation', 'spatial'))
    .sort('cropland_F1', false)
    .first()
);

Map.addLayer(label, {min: 0, max: 1, palette: CLASS_PALETTE}, 'label cropland ' + targetYearText, false);
Map.addLayer(probability, {min: 0, max: 1, palette: ['FFFFFF', 'E49635']}, 'cropland probability ' + targetYearText, false);
Map.addLayer(prediction, {min: 0, max: 1, palette: CLASS_PALETTE}, 'prediction cropland ' + targetYearText, true);
Map.addLayer(randomResult.trainPoints, {color: '00FF00'}, 'random train points', false);
Map.addLayer(randomResult.testPoints, {color: 'FF0000'}, 'random test points', false);
Map.addLayer(spatialResult.trainPoints, {color: '0000FF'}, 'spatial train points', false);
Map.addLayer(spatialResult.testPoints, {color: 'FFFF00'}, 'spatial test points', false);

var filePrefix = 'gaoyou_cropland_binary_' + MODEL_NAME + '_' + targetYearText;

Export.table.toDrive({
  collection: resultTable,
  description: 'Gaoyou_cropland_binary_accuracy_' + MODEL_NAME + '_' + targetYearText,
  folder: DRIVE_FOLDER,
  fileNamePrefix: filePrefix + '_accuracy',
  fileFormat: 'CSV',
  selectors: [
    'record_type',
    'year',
    'model',
    'validation',
    'feature_count',
    'train_sample_count',
    'test_sample_count',
    'overall_accuracy',
    'kappa',
    'macro_f1',
    'threshold',
    'class_code',
    'class_name',
    'PA',
    'UA',
    'F1',
    'non_cropland_PA',
    'non_cropland_UA',
    'non_cropland_F1',
    'cropland_PA',
    'cropland_UA',
    'cropland_F1',
    'actual_code',
    'actual_name',
    'predicted_code',
    'predicted_name',
    'sample_count',
    'cropland_area_m2',
    'cropland_area_ha',
    'cropland_area_km2',
    'valid_area_m2',
    'cropland_ratio'
  ]
});

Export.table.toDrive({
  collection: areaTable,
  description: 'Gaoyou_cropland_binary_area_' + MODEL_NAME + '_' + targetYearText,
  folder: DRIVE_FOLDER,
  fileNamePrefix: filePrefix + '_area',
  fileFormat: 'CSV',
  selectors: [
    'record_type',
    'year',
    'model',
    'validation',
    'threshold',
    'cropland_area_m2',
    'cropland_area_ha',
    'cropland_area_km2',
    'valid_area_m2',
    'cropland_ratio'
  ]
});

if (EXPORT_CROPLAND_TIF) {
  Export.image.toDrive({
    image: prediction.clip(region),
    description: 'Gaoyou_cropland_binary_' + MODEL_NAME + '_' + targetYearText,
    folder: DRIVE_FOLDER,
    fileNamePrefix: filePrefix + '_10m',
    region: region,
    scale: SCALE,
    crs: EXPORT_CRS,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });
}

// ============================================================================
// 7. 可选：多年预测耕地面积表
// ============================================================================

if (EXPORT_MULTI_YEAR_AREA_TABLE) {
  var multiYearAreaRows = [];
  for (var yearIndex = 0; yearIndex < YEAR_LIST.length; yearIndex++) {
    var year = YEAR_LIST[yearIndex];
    var yearLabel = makeBinaryLabel(year);
    multiYearAreaRows.push(ee.Feature(evaluateYear(year, MODEL_NAME, 'spatial').area.first()));
    multiYearAreaRows.push(referenceArea(yearLabel, year));
  }

  Export.table.toDrive({
    collection: ee.FeatureCollection(multiYearAreaRows),
    description: 'Gaoyou_cropland_binary_multi_year_area_' + MODEL_NAME,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'gaoyou_cropland_binary_multi_year_area_' + MODEL_NAME,
    fileFormat: 'CSV',
    selectors: [
      'record_type',
      'year',
      'model',
      'validation',
      'threshold',
      'cropland_area_m2',
      'cropland_area_ha',
      'cropland_area_km2',
      'valid_area_m2',
      'cropland_ratio'
    ]
  });
}
