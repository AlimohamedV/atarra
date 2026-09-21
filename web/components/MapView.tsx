"use client";

/**
 * MapLibre map with the rendered index composite as an image overlay.
 *
 * Four decisions worth noting:
 *
 * 1. The overlay is placed using the four grid corners the API returns, not a
 *    lon/lat bounding box. A UTM grid rectangle becomes a rotated quadrilateral in
 *    lon/lat, so a bbox would stretch the overlay off the imagery it describes.
 *
 * 2. All basemaps are keyless (Esri World Imagery, OpenStreetMap, or none). The
 *    project's premise is zero cost, and a Mapbox token would quietly break that.
 *
 * 3. There is exactly one place that mutates map state (`sync`), and it runs from
 *    map events (`load`, `styledata`, `idle`) as well as on prop changes. This
 *    matters more than it looks. React can deliver props before the map exists or
 *    before its style has finished loading, and `setStyle` tears down every custom
 *    source and layer *asynchronously*. A component that only reacts to props or
 *    only to a `ready` flag will therefore intermittently lose layers or never
 *    apply its initial camera: gating on `ready` alone shipped a map that opened
 *    at the default view instead of the study area, because the fit was requested
 *    on a render where the flag had not yet flipped.
 *
 * 4. Both the camera fit and the overlay are keyed by a serialised value that is
 *    remembered per map instance. Re-applying is cheap and idempotent, so an event
 *    storm cannot cause repeated animated camera moves, while a genuinely new fit
 *    is guaranteed to be applied no matter which event arrives first.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import maplibregl, { Map as MlMap, StyleSpecification } from "maplibre-gl";
import type { ImageCorners } from "@/lib/api";

export type BasemapKey = "satellite" | "streets" | "none";

const BASEMAPS: Record<BasemapKey, { tiles: string[]; attribution: string }> = {
  satellite: {
    tiles: [
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    ],
    attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
  },
  streets: {
    tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
    attribution: "© OpenStreetMap contributors",
  },
  none: { tiles: [], attribution: "No basemap — overlay only" },
};

const OVERLAY_SOURCE = "atarra-overlay";
const OVERLAY_LAYER = "atarra-overlay-layer";
const AOI_SOURCE = "atarra-aoi";
const AOI_LINE_LAYER = `${AOI_SOURCE}-line`;

function baseStyle(basemap: BasemapKey): StyleSpecification {
  const style: StyleSpecification = {
    version: 8,
    sources: {},
    layers: [
      {
        id: "background",
        type: "background",
        paint: { "background-color": "#0b1220" },
      },
    ],
  };

  if (basemap !== "none") {
    style.sources = {
      basemap: {
        type: "raster",
        tiles: BASEMAPS[basemap].tiles,
        tileSize: 256,
        attribution: BASEMAPS[basemap].attribution,
      },
    };
    style.layers.push({
      id: "basemap",
      type: "raster",
      source: "basemap",
      // Slightly translucent so the index overlay reads clearly on top of it.
      paint: { "raster-opacity": 0.8 },
    });
  }

  return style;
}

export type MapViewProps = {
  /** [topLeft, topRight, bottomRight, bottomLeft] as [lon, lat], or null. */
  corners: ImageCorners | null;
  overlayUrl: string | null;
  opacity: number;
  visible: boolean;
  basemap: BasemapKey;
  aoiGeometry: { type: "Polygon"; coordinates: number[][][] } | null;
  fitBounds: [number, number, number, number] | null;
  onError?: (message: string) => void;
};

export default function MapView({
  corners,
  overlayUrl,
  opacity,
  visible,
  basemap,
  aoiGeometry,
  fitBounds,
  onError,
}: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MlMap | null>(null);
  const [ready, setReady] = useState(false);

  // Latest props, readable from event handlers that are registered once.
  const propsRef = useRef({
    corners,
    overlayUrl,
    opacity,
    visible,
    aoiGeometry,
    fitBounds,
  });
  propsRef.current = { corners, overlayUrl, opacity, visible, aoiGeometry, fitBounds };

  // Identity of what is currently attached, so a change is detected even after a
  // style reload has wiped the layers. Both are reset whenever the map is rebuilt.
  const appliedOverlayRef = useRef<string | null>(null);
  /** The camera fit the app is currently asking for. */
  const desiredFitRef = useRef<string | null>(null);
  /** That fit, stamped with the canvas size it was applied at. */
  const appliedFitRef = useRef<string | null>(null);
  /** True once the user has moved the camera themselves, so we stop re-fitting. */
  const userMovedRef = useRef(false);
  /** Guards against our own animated fit counting as a user gesture. */
  const programmaticRef = useRef(false);

  /** Ensure the AOI outline, index overlay, and camera all match current props. */
  const sync = useCallback(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    const current = propsRef.current;

    // --- AOI outline ---
    if (current.aoiGeometry) {
      const data = {
        type: "Feature" as const,
        properties: {},
        geometry: current.aoiGeometry,
      };
      const existing = map.getSource(AOI_SOURCE) as maplibregl.GeoJSONSource | undefined;
      if (existing) {
        existing.setData(data);
      } else {
        map.addSource(AOI_SOURCE, { type: "geojson", data });
        map.addLayer({
          id: `${AOI_SOURCE}-fill`,
          type: "fill",
          source: AOI_SOURCE,
          paint: { "fill-color": "#38bdf8", "fill-opacity": 0.04 },
        });
        map.addLayer({
          id: AOI_LINE_LAYER,
          type: "line",
          source: AOI_SOURCE,
          paint: { "line-color": "#38bdf8", "line-width": 1.5, "line-dasharray": [3, 2] },
        });
      }
    }

    // --- index overlay ---
    // Keyed on the URL *and* the corners: an image source bakes its coordinates in
    // at creation, so a new grid with the same URL would otherwise keep rendering
    // at the old location.
    const overlayKey =
      current.corners && current.overlayUrl
        ? `${current.overlayUrl}|${current.corners.flat().join(",")}`
        : null;
    const sourceExists = Boolean(map.getSource(OVERLAY_SOURCE));

    if (!overlayKey) {
      if (map.getLayer(OVERLAY_LAYER)) map.removeLayer(OVERLAY_LAYER);
      if (sourceExists) map.removeSource(OVERLAY_SOURCE);
      appliedOverlayRef.current = null;
    } else if (sourceExists && appliedOverlayRef.current !== overlayKey) {
      // Replace rather than mutate: `updateImage` on a source whose image has not
      // finished loading is unreliable, and a brief re-draw is a far better failure
      // mode than a permanently stale overlay.
      if (map.getLayer(OVERLAY_LAYER)) map.removeLayer(OVERLAY_LAYER);
      map.removeSource(OVERLAY_SOURCE);
    }

    if (overlayKey && !map.getSource(OVERLAY_SOURCE)) {
      map.addSource(OVERLAY_SOURCE, {
        type: "image",
        url: current.overlayUrl as string,
        coordinates: current.corners as ImageCorners,
      });
      map.addLayer({
        id: OVERLAY_LAYER,
        type: "raster",
        source: OVERLAY_SOURCE,
        paint: {
          "raster-opacity": current.visible ? current.opacity : 0,
          "raster-fade-duration": 0,
          // Index values are continuous, so linear interpolation is right here --
          // unlike a segmentation mask, which must stay blocky.
          "raster-resampling": "linear",
        },
      });
      appliedOverlayRef.current = overlayKey;
    } else if (overlayKey && map.getLayer(OVERLAY_LAYER)) {
      map.setPaintProperty(OVERLAY_LAYER, "raster-opacity", current.visible ? current.opacity : 0);
      appliedOverlayRef.current = overlayKey;
    }

    // Keep the study-area outline on top. It is drawn after the overlay only when
    // both are created in the same pass; a later overlay rebuild would otherwise
    // bury the dashed boundary under an 85%-opaque raster, hiding it completely.
    if (map.getLayer(AOI_LINE_LAYER)) {
      map.moveLayer(AOI_LINE_LAYER);
    }

    // --- camera ---
    // The fit is remembered together with the canvas size it was computed at,
    // because `fitBounds` with a zero-height container produces a meaningless zoom
    // that then sticks. Layout has not necessarily settled when the first props
    // arrive, so the stamp lets a corrected fit land once the size is real -- and
    // a later resize (a split pane, a rotated phone) re-fits instead of leaving the
    // study area half off-screen.
    if (!current.fitBounds) {
      desiredFitRef.current = null;
      appliedFitRef.current = null;
      return;
    }

    const key = current.fitBounds.join(",");
    if (desiredFitRef.current !== key) {
      // A different AOI supersedes wherever the user had navigated to.
      desiredFitRef.current = key;
      appliedFitRef.current = null;
      userMovedRef.current = false;
    }

    const canvas = map.getCanvas();
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width <= 0 || height <= 0) return; // too early; a later event will retry

    const stamp = `${key}@${width}x${height}`;
    if (appliedFitRef.current === stamp || userMovedRef.current) return;

    const [west, south, east, north] = current.fitBounds;
    appliedFitRef.current = stamp;
    programmaticRef.current = true;
    map.once("moveend", () => {
      programmaticRef.current = false;
    });
    map.fitBounds(
      [
        [west, south],
        [east, north],
      ],
      { padding: 40, duration: 800 },
    );
  }, []);

  // --- create the map once -------------------------------------------------
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: baseStyle(basemap),
      center: [30.85, 31.5], // Lake Burullus
      zoom: 9,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

    map.on("load", () => setReady(true));
    // Remember manual navigation so a resize does not yank the view back.
    for (const event of ["dragstart", "zoomstart", "rotatestart"] as const) {
      map.on(event, () => {
        if (!programmaticRef.current) userMovedRef.current = true;
      });
    }
    map.on("error", (event) => {
      const message = (event as { error?: Error }).error?.message;
      // Tile and generic network failures are noisy, transient, and non-fatal --
      // the basemap and overlay both have visible fallbacks. Only surface errors
      // that name something specific enough to act on.
      if (!message) return;
      if (/tile|fetch|network|abort|timeout/i.test(message)) return;
      onError?.(message);
    });

    mapRef.current = map;

    // MapLibre does not watch its container, so a flex or responsive layout
    // changing the pane size leaves the canvas at its old dimensions. Observing it
    // is what makes the `resize` event -- and therefore the re-fit -- happen.
    const observer = new ResizeObserver(() => map.resize());
    observer.observe(containerRef.current);

    return () => {
      observer.disconnect();
      map.remove();
      mapRef.current = null;
      appliedOverlayRef.current = null;
      appliedFitRef.current = null;
      setReady(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- re-apply after every style load ------------------------------------
  // `idle` also catches the case where a source finishes loading after the props
  // have already been delivered, which is where a prop-only effect gives up.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const handler = () => sync();
    map.on("load", handler);
    map.on("styledata", handler);
    map.on("idle", handler);
    // Fires whenever the container is resized, which is our cue to re-fit.
    map.on("resize", handler);
    return () => {
      map.off("load", handler);
      map.off("styledata", handler);
      map.off("idle", handler);
      map.off("resize", handler);
    };
  }, [sync]);

  // --- swap the basemap, but only when it actually changes -----------------
  const previousBasemap = useRef(basemap);
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    if (previousBasemap.current === basemap) return;
    previousBasemap.current = basemap;
    // Wipes custom sources and layers; the `styledata` handler re-applies them.
    map.setStyle(baseStyle(basemap));
  }, [basemap, ready]);

  // --- react to prop changes ----------------------------------------------
  // Deliberately *not* gated on `ready`: `sync` checks style readiness itself, and
  // the map events cover the case where props arrive first.
  useEffect(() => {
    sync();
  }, [corners, overlayUrl, aoiGeometry, fitBounds, opacity, visible, sync, ready]);

  return <div ref={containerRef} className="h-full w-full" />;
}
