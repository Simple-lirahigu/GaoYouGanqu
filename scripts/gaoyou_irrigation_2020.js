/**
 * 高邮灌区 2020 年灌溉稻田初步识别（GEE JavaScript）
 *
 * 目标：
 * 1. 使用 2020 年土地利用数据约束耕地范围；
 * 2. 使用 Sentinel-1/2 时序识别“移栽期有水 + 生长期植被旺盛”的稻田；
 * 3. 输出灌溉稻田分布、面积统计和诊断图层。
 *
 * 重要限制：
 * - 本脚本识别的是具有水稻物候特征的灌溉耕地，不等同于所有灌溉农田。
 * - 未使用独立地面样本，当前结果不能直接声称具有某一分类精度。
 * - 必须先根据 Console 中的直方图确认 LANDUSE_BAND 和 CROPLAND_CODES。
 */

// ============================================================================
// 0. 参数区：首次运行时重点检查这里
// ============================================================================

var aoi = ee.FeatureCollection('projects/ee-yangsimple237/assets/GYBJ');
var landuse = ee.Image('projects/ee-yangsimple237/assets/2020tudi');

// 如果土地利用影像的分类波段不是第一个波段，请改成实际波段名。
// 留空时自动使用第一个波段。
var LANDUSE_BAND = null;

// 必须按 Console 输出的类别编码修改。
// 下面的 1 只是常见“耕地”占位值，不代表 2020tudi 的真实编码。
var CROPLAND_CODES = [1];

// 如果已知土地利用中的水体类别，可填写，例如 [5]；未知可保留空数组。
var WATER_CODES = [];

var START_DATE = '2020-01-01';
var END_DATE = '2021-01-01';
var SCALE = 10;
var EXPORT_CRS = 'EPSG:32650'; // 高邮附近 UTM 50N

// 高邮单季稻常见物候窗口。若当地种植历与此不同，应调整。
var FLOOD_START = '2020-05-01';
var FLOOD_END = '2020-07-01';
var GROWTH_START = '2020-07-01';
var GROWTH_END = '2020-10-01';

// 初始阈值，需要结合样本或高分辨率影像校准。
var FLOOD_SCORE_MIN = 2;       // 4项水分/雷达证据中至少满足2项
var PEAK_NDVI_MIN = 0.55;      // 盛夏植被峰值
var NDVI_AMPLITUDE_MIN = 0.25; // 年内植被振幅
var MIN_PATCH_PIXELS = 8;      // 去除小于约0.08 ha的孤立斑块
var PERMANENT_WATER_OCCURRENCE = 50;

var region = aoi.geometry();
var landuseClass = LANDUSE_BAND
  ? landuse.select(LANDUSE_BAND).rename('landuse')
  : landuse.select([0]).rename('landuse');

Map.centerObject(aoi, 10);
Map.addLayer(aoi.style({color: 'FF0000', fillColor: '00000000'}), {}, 'AOI');
Map.addLayer(landuseClass.clip(region), {}, '2020土地利用（原始）', false);

print('土地利用影像波段：', landuse.bandNames());
print(
  '研究区土地利用类别直方图：',
  landuseClass.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: region,
    scale: SCALE,
    maxPixels: 1e10,
    tileScale: 4
  })
);

function maskFromCodes(image, codes) {
  var mask = ee.Image(0);
  codes.forEach(function(code) {
    mask = mask.or(image.eq(code));
  });
  return mask;
}

var croplandMask = maskFromCodes(landuseClass, CROPLAND_CODES)
  .selfMask()
  .rename('cropland');
var landuseWaterMask = WATER_CODES.length > 0
  ? maskFromCodes(landuseClass, WATER_CODES)
  : ee.Image(0);

Map.addLayer(croplandMask.clip(region), {palette: ['F4A261']}, '耕地掩膜');

// ============================================================================
// 1. Sentinel-2：云掩膜与指数
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
  var lswi = image.normalizedDifference(['nir', 'swir1']).rename('LSWI');
  var mndwi = image.normalizedDifference(['green', 'swir1']).rename('MNDWI');
  var ndre = image.normalizedDifference(['nirNarrow', 're1']).rename('NDRE');
  var evi = image.expression(
    '2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)',
    {
      nir: image.select('nir'),
      red: image.select('red'),
      blue: image.select('blue')
    }
  ).rename('EVI');

  return image.addBands([ndvi, lswi, mndwi, ndre, evi]);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
  .map(maskS2)
  .map(addS2Indices);

print('2020年 Sentinel-2 影像数量：', s2.size());

var s2Flood = s2.filterDate(FLOOD_START, FLOOD_END);
var s2Growth = s2.filterDate(GROWTH_START, GROWTH_END);

var floodLswiMax = s2Flood.select('LSWI').max().rename('flood_lswi_max');
var floodMndwiMax = s2Flood.select('MNDWI').max().rename('flood_mndwi_max');
var floodNdviMedian = s2Flood.select('NDVI').median().rename('flood_ndvi_median');
var growthNdviMax = s2Growth.select('NDVI').max().rename('growth_ndvi_max');
var growthEviMax = s2Growth.select('EVI').max().rename('growth_evi_max');
var growthNdreMax = s2Growth.select('NDRE').max().rename('growth_ndre_max');
var annualNdviMax = s2.select('NDVI').max();
var annualNdviMin = s2.select('NDVI').min();
var ndviAmplitude = annualNdviMax.subtract(annualNdviMin).rename('ndvi_amplitude');

// 水稻移栽期常见光学证据：水分指数接近或超过植被指数。
var lswiVsNdvi = floodLswiMax
  .add(0.05)
  .gte(floodNdviMedian)
  .rename('lswi_vs_ndvi');

// ============================================================================
// 2. Sentinel-1：雷达积水与时序变化
// ============================================================================

function prepareS1(image) {
  var vv = image.select('VV');
  var vh = image.select('VH');
  var ratio = vv.subtract(vh).rename('VV_minus_VH');
  return vv.rename('VV')
    .addBands(vh.rename('VH'))
    .addBands(ratio)
    .copyProperties(image, ['system:time_start']);
}

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(region)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select(['VV', 'VH'])
  .map(prepareS1);

print('2020年 Sentinel-1 影像数量（降轨）：', s1.size());

var s1Flood = s1.filterDate(FLOOD_START, FLOOD_END);
var s1Growth = s1.filterDate(GROWTH_START, GROWTH_END);

var floodVvMin = s1Flood.select('VV').min().rename('flood_vv_min');
var floodVhMin = s1Flood.select('VH').min().rename('flood_vh_min');
var growthVhMedian = s1Growth.select('VH').median().rename('growth_vh_median');
var vhIncrease = growthVhMedian.subtract(floodVhMin).rename('vh_increase');

// 这些阈值是初始经验值，必须用高邮样本检查。
var vvFloodEvidence = floodVvMin.lt(-15.5).rename('vv_flood_evidence');
var vhFloodEvidence = floodVhMin.lt(-23.0).rename('vh_flood_evidence');

// ============================================================================
// 3. 灌溉稻田判定
// ============================================================================

// 四项移栽期水分证据：LSWI/NDVI、MNDWI、VV低后向散射、VH低后向散射。
var floodScore = lswiVsNdvi
  .add(floodMndwiMax.gt(-0.05))
  .add(vvFloodEvidence)
  .add(vhFloodEvidence)
  .rename('flood_score');

var vegetationCondition = growthNdviMax.gte(PEAK_NDVI_MIN)
  .and(ndviAmplitude.gte(NDVI_AMPLITUDE_MIN));

// JRC永久水体用于排除河流、湖泊和部分长期养殖水面。
var waterOccurrence = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence');
var permanentWater = waterOccurrence
  .gte(PERMANENT_WATER_OCCURRENCE)
  .or(landuseWaterMask);

var irrigationRaw = floodScore.gte(FLOOD_SCORE_MIN)
  .and(vegetationCondition)
  .and(croplandMask.unmask(0).eq(1))
  .and(permanentWater.not())
  .rename('irrigated_rice_2020');

// 去除极小孤立像元。注意：这不是田块边界提取。
var connectedPixels = irrigationRaw.selfMask().connectedPixelCount(100, true);
var irrigation = irrigationRaw
  .updateMask(connectedPixels.gte(MIN_PATCH_PIXELS))
  .selfMask()
  .toByte()
  .rename('irrigated_rice_2020');

// ============================================================================
// 4. 显示与诊断
// ============================================================================

var s2Rgb = s2.filterDate('2020-07-01', '2020-09-30').median();
Map.addLayer(
  s2Rgb.clip(region),
  {bands: ['red', 'green', 'blue'], min: 0.02, max: 0.35},
  'Sentinel-2 夏季RGB',
  false
);
Map.addLayer(
  floodScore.updateMask(croplandMask).clip(region),
  {min: 0, max: 4, palette: ['FFFFFF', 'FFF3B0', 'F8961E', 'D00000']},
  '移栽期水分证据得分',
  false
);
Map.addLayer(
  growthNdviMax.updateMask(croplandMask).clip(region),
  {min: 0.2, max: 0.9, palette: ['8C510A', 'DFC27D', '80CDC1', '01665E']},
  '生长期最大NDVI',
  false
);
Map.addLayer(
  vhIncrease.updateMask(croplandMask).clip(region),
  {min: 0, max: 12, palette: ['440154', '21908C', 'FDE725']},
  'VH由移栽期到生长期增量',
  false
);
Map.addLayer(
  permanentWater.selfMask().clip(region),
  {palette: ['0000FF']},
  '排除的永久水体',
  false
);
Map.addLayer(
  irrigation.clip(region),
  {min: 1, max: 1, palette: ['00C853']},
  '2020灌溉稻田初识别',
  true
);

// ============================================================================
// 5. 面积统计
// ============================================================================

var irrigationAreaImage = irrigation
  .multiply(ee.Image.pixelArea())
  .rename('area_m2');

var irrigationArea = irrigationAreaImage.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e10,
  tileScale: 4
});

var areaM2 = ee.Number(irrigationArea.get('area_m2'));
print('初识别灌溉稻田面积（平方米）：', areaM2);
print('初识别灌溉稻田面积（公顷）：', areaM2.divide(1e4));
print('初识别灌溉稻田面积（平方千米）：', areaM2.divide(1e6));

var areaSummary = ee.FeatureCollection([
  ee.Feature(null, {
    year: 2020,
    target: 'irrigated_rice',
    area_m2: areaM2,
    area_ha: areaM2.divide(1e4),
    area_km2: areaM2.divide(1e6),
    flood_start: FLOOD_START,
    flood_end: FLOOD_END,
    growth_start: GROWTH_START,
    growth_end: GROWTH_END,
    flood_score_min: FLOOD_SCORE_MIN,
    peak_ndvi_min: PEAK_NDVI_MIN,
    ndvi_amplitude_min: NDVI_AMPLITUDE_MIN,
    min_patch_pixels: MIN_PATCH_PIXELS
  })
]);

// ============================================================================
// 6. 导出
// ============================================================================

Export.image.toDrive({
  image: irrigation.unmask(0).clip(region),
  description: 'Gaoyou_irrigated_rice_2020_10m',
  folder: 'GEE_Gaoyou_Irrigation',
  fileNamePrefix: 'gaoyou_irrigated_rice_2020_10m',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});

Export.image.toDrive({
  image: ee.Image.cat([
    floodScore,
    floodLswiMax,
    floodMndwiMax,
    floodVvMin,
    floodVhMin,
    growthNdviMax,
    growthEviMax,
    growthNdreMax,
    ndviAmplitude,
    vhIncrease
  ]).updateMask(croplandMask).clip(region),
  description: 'Gaoyou_irrigation_diagnostics_2020',
  folder: 'GEE_Gaoyou_Irrigation',
  fileNamePrefix: 'gaoyou_irrigation_diagnostics_2020',
  region: region,
  scale: SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});

Export.table.toDrive({
  collection: areaSummary,
  description: 'Gaoyou_irrigated_rice_area_2020',
  folder: 'GEE_Gaoyou_Irrigation',
  fileNamePrefix: 'gaoyou_irrigated_rice_area_2020',
  fileFormat: 'CSV'
});
