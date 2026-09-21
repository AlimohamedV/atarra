/**
 * Typed client for the ATARRA API.
 *
 * The API base URL is configurable so the dashboard can point at a local server,
 * a LAN address, or a deployment without a rebuild-time change to the code.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://localhost:8000";

export type StudyArea = {
  key: string;
  name: string;
  role: "wetland" | "canal";
  crs: string;
  bbox_wgs84: [number, number, number, number];
  geometry: { type: "Polygon"; coordinates: number[][][] };
};

export type SceneDate = {
  date: string;
  scenes: number;
  tiles: string[];
  best_cloud_cover: number | null;
};

export type IndexStats = {
  min: number;
  median: number;
  max: number;
  p10: number;
  p90: number;
};

export type CompositeSummary = {
  area: string;
  area_name: string;
  date: string;
  window_days: number;
  grid: {
    crs: string;
    width: number;
    height: number;
    gsd: number;
    bounds_wgs84: number[];
  };
  coverage: number;
  scenes: {
    id: string;
    acquired: string;
    platform: string;
    cloud_cover: number | null;
    tile: string | null;
  }[];
  indices: Record<string, IndexStats>;
  reflectance: {
    n: number;
    negative_fraction: number;
    median: number;
    max: number;
  };
  warnings: string[];
  from_cache: boolean;
  /** [topLeft, topRight, bottomRight, bottomLeft] as [lon, lat], for MapLibre. */
  image_corners: ImageCorners;
  available_indices: string[];
};

/**
 * The four grid corners as [lon, lat], ordered top-left, top-right,
 * bottom-right, bottom-left.
 *
 * Modelled as a fixed 4-tuple rather than `[number, number][]` because that is
 * genuinely what it is: a quadrilateral, not an arbitrary list. Encoding the
 * arity in the type is what stops a truncated array reaching MapLibre and
 * producing a silently distorted overlay.
 */
export type ImageCorners = [
  [number, number],
  [number, number],
  [number, number],
  [number, number],
];

export type LegendStop = { value: number; color: string };

export type Legend = {
  index: string;
  vmin: number;
  vmax: number;
  stops: LegendStop[];
};

export type Health = {
  status: string;
  version: string;
  imagery_source: string;
  collection: string;
  reflectance: {
    mode: string;
    scale: number;
    offset: number;
    max_negative_fraction: number;
  };
  bands_8: string[];
  indices: string[];
  cache: {
    entries: number;
    megabytes: number;
    max_bytes: number;
    usage_fraction: number | null;
  };
};

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) {
    // Surface the API's own message where there is one: "no usable scene for
    // this date" is far more useful to a user than a bare status code.
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => getJson<Health>("/health"),

  areas: () =>
    getJson<{ count: number; areas: StudyArea[] }>("/areas").then((r) => r.areas),

  scenes: (area: string, months = 12) => {
    const end = new Date();
    const start = new Date(end);
    start.setMonth(start.getMonth() - months);
    const iso = (d: Date) => d.toISOString().slice(0, 10);
    return getJson<{ dates: SceneDate[] }>(
      `/areas/${area}/scenes?start=${iso(start)}&end=${iso(end)}&max_cloud_cover=20`,
    ).then((r) => r.dates);
  },

  summary: (area: string, date: string, index: string, windowDays: number) =>
    getJson<CompositeSummary>(
      `/areas/${area}/summary?date=${date}&window_days=${windowDays}&gsd=30&indices=${index}`,
    ),

  legend: (index: string) => getJson<Legend>(`/legend/${index}`),

  /** PNG overlay URL. The parameters make it naturally cache-friendly. */
  renderUrl: (area: string, date: string, index: string, windowDays: number) =>
    `${API_BASE}/areas/${area}/render.png?date=${date}&index=${index}&window_days=${windowDays}&gsd=30`,

  trueColorUrl: (area: string, date: string, windowDays: number) =>
    `${API_BASE}/areas/${area}/truecolor.png?date=${date}&window_days=${windowDays}&gsd=30`,
};

export const INDEX_LABELS: Record<string, string> = {
  ndvi: "NDVI",
  ndwi: "NDWI",
  ndre: "NDRE",
  ndmi: "NDMI",
};

export const INDEX_DESCRIPTIONS: Record<string, string> = {
  ndvi: "Vegetation vigour — NIR vs red",
  ndwi: "Open water — green vs NIR",
  ndre: "Red-edge canopy structure — the reed vs crop discriminator",
  ndmi: "Canopy moisture — NIR vs SWIR",
};
