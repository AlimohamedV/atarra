"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import {
  api,
  API_BASE,
  INDEX_DESCRIPTIONS,
  INDEX_LABELS,
  type CompositeSummary,
  type Health,
  type Legend as LegendData,
  type SceneDate,
  type StudyArea,
} from "@/lib/api";
import type { BasemapKey } from "@/components/MapView";

// MapLibre touches `window` at import time, so the map must never render on the
// server. Without this the page throws during the Next.js build.
const MapView = dynamic(() => import("@/components/MapView"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-sm text-slate-400">
      Loading map…
    </div>
  ),
});

const INDEXES = ["ndvi", "ndwi", "ndre", "ndmi"] as const;

export default function Dashboard() {
  const [health, setHealth] = useState<Health | null>(null);
  const [areas, setAreas] = useState<StudyArea[]>([]);
  const [areaKey, setAreaKey] = useState<string>("burullus");
  const [dates, setDates] = useState<SceneDate[]>([]);
  const [date, setDate] = useState<string>("");
  const [index, setIndex] = useState<string>("ndvi");
  const [windowDays, setWindowDays] = useState(3);
  const [summary, setSummary] = useState<CompositeSummary | null>(null);
  const [legend, setLegend] = useState<LegendData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opacity, setOpacity] = useState(0.85);
  const [visible, setVisible] = useState(true);
  const [basemap, setBasemap] = useState<BasemapKey>("satellite");
  const [showTrueColor, setShowTrueColor] = useState(false);

  const area = useMemo(() => areas.find((a) => a.key === areaKey) ?? null, [areas, areaKey]);

  // --- initial load -------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [h, a] = await Promise.all([api.health(), api.areas()]);
        if (cancelled) return;
        setHealth(h);
        setAreas(a);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // --- available dates when the area changes ------------------------------
  useEffect(() => {
    if (!areaKey) return;
    let cancelled = false;
    setSummary(null);
    setDates([]);
    setDate("");
    (async () => {
      try {
        const found = await api.scenes(areaKey, 12);
        if (cancelled) return;
        setDates(found);
        // Preselect the clearest recent acquisition rather than the newest: the
        // newest may be a cloudy scene that is technically available but useless.
        const best = [...found]
          .filter((d) => d.best_cloud_cover !== null)
          .sort((a, b) => (a.best_cloud_cover ?? 99) - (b.best_cloud_cover ?? 99))[0];
        setDate(best?.date ?? found.at(-1)?.date ?? "");
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [areaKey]);

  // --- legend -------------------------------------------------------------
  useEffect(() => {
    api.legend(index).then(setLegend).catch(() => setLegend(null));
  }, [index]);

  // --- composite summary --------------------------------------------------
  const loadSummary = useCallback(async () => {
    if (!areaKey || !date) return;
    setLoading(true);
    setError(null);
    try {
      setSummary(await api.summary(areaKey, date, index, windowDays));
    } catch (e) {
      setSummary(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [areaKey, date, index, windowDays]);

  useEffect(() => {
    void loadSummary();
  }, [loadSummary]);

  const overlayUrl = useMemo(() => {
    if (!areaKey || !date) return null;
    return showTrueColor
      ? api.trueColorUrl(areaKey, date, windowDays)
      : api.renderUrl(areaKey, date, index, windowDays);
  }, [areaKey, date, index, windowDays, showTrueColor]);

  const stats = summary?.indices?.[index];

  return (
    <div className="flex h-screen flex-col bg-slate-950 text-slate-100">
      {/* ---------------- header ---------------- */}
      <header className="flex flex-wrap items-center gap-4 border-b border-slate-800 px-5 py-3">
        <div className="flex items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight">ATARRA</h1>
          <span className="hidden text-xs text-slate-400 sm:inline">
            Aquatic invasive weed monitoring · Nile Delta
          </span>
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-3 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-slate-400">Zone</span>
            <select
              value={areaKey}
              onChange={(e) => setAreaKey(e.target.value)}
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1"
            >
              {areas.map((a) => (
                <option key={a.key} value={a.key}>
                  {a.name}
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2">
            <span className="text-slate-400">Date</span>
            <select
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1"
              disabled={!dates.length}
            >
              {!dates.length && <option value="">no imagery</option>}
              {dates.map((d) => (
                <option key={d.date} value={d.date}>
                  {d.date} · {d.best_cloud_cover?.toFixed(1) ?? "?"}% cloud
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2">
            <span className="text-slate-400" title="Days either side used to fill MGRS tile gaps">
              ±days
            </span>
            <input
              type="number"
              min={0}
              max={15}
              value={windowDays}
              onChange={(e) => setWindowDays(Math.max(0, Math.min(15, Number(e.target.value))))}
              className="w-16 rounded border border-slate-700 bg-slate-900 px-2 py-1"
            />
          </label>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* ---------------- sidebar ---------------- */}
        <aside className="flex w-80 shrink-0 flex-col gap-4 overflow-y-auto border-r border-slate-800 p-4 text-sm">
          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Spectral index
            </h2>
            <div className="grid grid-cols-2 gap-2">
              {INDEXES.map((key) => (
                <button
                  key={key}
                  onClick={() => {
                    setIndex(key);
                    setShowTrueColor(false);
                  }}
                  title={INDEX_DESCRIPTIONS[key]}
                  className={`rounded border px-2 py-1.5 text-left transition ${
                    index === key && !showTrueColor
                      ? "border-sky-400 bg-sky-500/15 text-sky-200"
                      : "border-slate-700 bg-slate-900 hover:border-slate-500"
                  }`}
                >
                  {INDEX_LABELS[key]}
                </button>
              ))}
            </div>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">
              {INDEX_DESCRIPTIONS[index]}
            </p>
            <button
              onClick={() => setShowTrueColor((v) => !v)}
              className={`mt-2 w-full rounded border px-2 py-1.5 transition ${
                showTrueColor
                  ? "border-sky-400 bg-sky-500/15 text-sky-200"
                  : "border-slate-700 bg-slate-900 hover:border-slate-500"
              }`}
            >
              {showTrueColor ? "Showing true colour" : "Show true colour"}
            </button>
          </section>

          <section className="space-y-3">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Display
            </h2>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={visible}
                onChange={(e) => setVisible(e.target.checked)}
              />
              <span>Overlay visible</span>
            </label>
            <label className="block">
              <span className="text-slate-400">Opacity {Math.round(opacity * 100)}%</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                className="w-full"
              />
            </label>
            <label className="flex items-center gap-2">
              <span className="text-slate-400">Basemap</span>
              <select
                value={basemap}
                onChange={(e) => setBasemap(e.target.value as BasemapKey)}
                className="rounded border border-slate-700 bg-slate-900 px-2 py-1"
              >
                <option value="satellite">Satellite</option>
                <option value="streets">Streets</option>
                <option value="none">None</option>
              </select>
            </label>
          </section>

          {/* legend */}
          {legend && !showTrueColor && (
            <section>
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                {INDEX_LABELS[legend.index]} scale
              </h2>
              <div
                className="h-4 w-full rounded"
                style={{
                  background: `linear-gradient(to right, ${legend.stops
                    .map((s) => s.color)
                    .join(", ")})`,
                }}
              />
              <div className="mt-1 flex justify-between text-xs text-slate-400">
                <span>{legend.vmin.toFixed(1)}</span>
                <span>{legend.vmax.toFixed(1)}</span>
              </div>
            </section>
          )}

          {/* statistics */}
          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Composite
            </h2>
            {loading && <p className="text-slate-400">Building composite…</p>}
            {error && (
              <p className="rounded border border-rose-900 bg-rose-950/50 p-2 text-xs text-rose-200">
                {error}
              </p>
            )}
            {summary && !loading && (
              <dl className="space-y-1 text-xs">
                <Row label="Coverage" value={`${(summary.coverage * 100).toFixed(1)}%`} />
                <Row label="Scenes" value={String(summary.scenes.length)} />
                <Row
                  label="Grid"
                  value={`${summary.grid.width}×${summary.grid.height} @ ${summary.grid.gsd} m`}
                />
                <Row label="CRS" value={summary.grid.crs} />
                {stats && (
                  <>
                    <Row label="Median" value={stats.median.toFixed(3)} />
                    <Row label="p10 – p90" value={`${stats.p10.toFixed(2)} … ${stats.p90.toFixed(2)}`} />
                  </>
                )}
                <Row
                  label="Negative refl."
                  value={`${(summary.reflectance.negative_fraction * 100).toFixed(1)}%`}
                />
                <Row label="Cache" value={summary.from_cache ? "hit" : "built"} />
              </dl>
            )}
            {summary?.warnings?.length ? (
              <ul className="mt-2 space-y-1">
                {summary.warnings.map((w) => (
                  <li
                    key={w}
                    className="rounded border border-amber-900 bg-amber-950/40 p-2 text-xs text-amber-200"
                  >
                    {w}
                  </li>
                ))}
              </ul>
            ) : null}
          </section>

          {health && (
            <section className="mt-auto border-t border-slate-800 pt-3 text-xs text-slate-500">
              <p>
                v{health.version} · {health.collection}
              </p>
              <p>
                reflectance: {health.reflectance.mode} (offset {health.reflectance.offset})
              </p>
              <p>
                cache {health.cache.megabytes.toFixed(0)} MB /{" "}
                {(health.cache.max_bytes / 1e6).toFixed(0)} MB
              </p>
              <p className="truncate" title={API_BASE}>
                API: {API_BASE}
              </p>
            </section>
          )}
        </aside>

        {/* ---------------- map ---------------- */}
        <main className="relative min-w-0 flex-1">
          <MapView
            corners={summary?.image_corners ?? null}
            overlayUrl={overlayUrl}
            opacity={opacity}
            visible={visible}
            basemap={basemap}
            aoiGeometry={area?.geometry ?? null}
            fitBounds={area?.bbox_wgs84 ?? null}
            onError={setError}
          />
          {!date && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <p className="rounded bg-slate-900/90 px-4 py-2 text-sm text-slate-300">
                No imagery available for this zone in the last 12 months.
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-slate-400">{label}</dt>
      <dd className="font-mono text-slate-200">{value}</dd>
    </div>
  );
}
