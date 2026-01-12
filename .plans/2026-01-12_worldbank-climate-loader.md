# Plan: World Bank Climate Data Loader

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a loader for World Bank Climate Data API to fetch historical and projected climate indicators (temperature, precipitation) with proper dimensional units. Integration with existing dimtensor loaders module.

---

## Background

The World Bank provides comprehensive climate data via their Climate Data API, including:
- Historical data (1901-2009 for countries, 1960-2009 for basins)
- Projected data from 15 Global Circulation Models (GCMs)
- Two emissions scenarios
- Temperature and precipitation at country/basin spatial scales

This loader will enable dimtensor users to easily access climate data with automatic unit handling (celsius, millimeters, meters) for scientific analysis.

---

## Approach

### Option A: REST API with direct requests
- Description: Use requests library to call World Bank Climate Data API directly
- Pros:
  - Simple, follows existing loader patterns (NASAExoplanetLoader)
  - No additional dependencies beyond requests
  - Full control over API parameters
  - Caching built-in via BaseLoader
- Cons:
  - Need to handle API error cases manually
  - API structure requires careful URL construction

### Option B: Use existing Python wrapper (wbpy)
- Description: Use wbpy library as abstraction layer
- Pros:
  - Higher-level interface
  - Error handling included
- Cons:
  - Additional dependency
  - May not be maintained (last update uncertain)
  - Less control over caching
  - Abstracts away details we need to understand

### Decision: Option A - Direct REST API

Use direct requests to World Bank Climate Data API. This matches the pattern used in NASAExoplanetLoader and PRISMClimateLoader, provides full control, and leverages existing BaseLoader caching infrastructure. The API is straightforward enough to not require a wrapper library.

---

## Implementation Steps

1. [ ] Create WorldBankClimateLoader class extending CSVLoader (supports multiple formats)
2. [ ] Implement load() method with parameters:
   - country_code: ISO3 country code (e.g., "USA", "KEN")
   - variable: "tas" (temperature) or "pr" (precipitation)
   - data_type: "cru" (historical), or GCM model names for projections
   - temporal_scale: "year" or "month"
   - start_year: optional filter
   - end_year: optional filter
3. [ ] Create helper method _build_api_url() to construct endpoint URLs
4. [ ] Implement _parse_climate_response() to handle JSON/XML responses
5. [ ] Map temperature data to celsius Unit with proper dimension
6. [ ] Map precipitation data to mm Unit (already defined in core.units)
7. [ ] Add support for basin-level queries (optional but included in API)
8. [ ] Handle API errors gracefully (invalid country codes, missing data)
9. [ ] Write comprehensive tests with mocked API responses
10. [ ] Update __init__.py to export WorldBankClimateLoader
11. [ ] Add docstrings with usage examples

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/datasets/loaders/climate.py | Add WorldBankClimateLoader class (~200 lines) |
| src/dimtensor/datasets/loaders/__init__.py | Export WorldBankClimateLoader |
| tests/datasets/loaders/test_climate.py | Create tests for WorldBankClimateLoader |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit tests with mocked API responses:
  - Test temperature data loading (historical CRU)
  - Test precipitation data loading (historical CRU)
  - Test country-level queries (USA, KEN, DEU)
  - Test basin-level queries
  - Test year/month temporal scales
  - Test error handling (invalid country code, network errors)
  - Test date range filtering
- [ ] Integration test with real API (marked as optional/slow):
  - Fetch real temperature data for one country
  - Verify DimArray units are correct
  - Verify caching works
- [ ] Manual verification:
  - Compare output with World Bank web portal data
  - Verify celsius and mm units display correctly

---

## Risks / Edge Cases

- **Risk**: World Bank API may change endpoints or deprecate v1 API
  - Mitigation: Use versioned endpoint (/v1/), document API version, add user-facing error if API changes

- **Risk**: API rate limiting or availability issues
  - Mitigation: Leverage BaseLoader caching, add retry logic with exponential backoff

- **Edge case**: Missing data for certain country/year combinations
  - Handling: Return NaN values in DimArray, document in docstring

- **Edge case**: Different data availability periods (1901-2009 vs 1960-2009)
  - Handling: Document in docstring, return available data range in response metadata

- **Risk**: API returns XML by default, JSON requires explicit .json extension
  - Mitigation: Always request .json format in URL construction

- **Edge case**: Basin IDs are numeric, not ISO codes
  - Handling: Add separate load_basin() method or basin_id parameter that changes URL structure

- **Edge case**: Projected data requires GCM model selection (15 options)
  - Handling: Add gcm_model parameter with validation, provide common model list in docstring

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass with >90% coverage
- [ ] WorldBankClimateLoader can fetch historical temperature data
- [ ] WorldBankClimateLoader can fetch historical precipitation data
- [ ] DimArrays have correct units (celsius, mm)
- [ ] Docstrings include clear examples
- [ ] Loader exported in __init__.py
- [ ] Error handling for invalid inputs
- [ ] Caching works correctly
- [ ] CONTINUITY.md updated

---

## Notes / Log

### API Endpoint Structure

Base URL: `http://climatedataapi.worldbank.org/climateweb/rest/v1`

**Country queries:**
```
/v1/country/{data_type}/{var}/{temporal_scale}/{ISO3}[.ext]
```

**Basin queries:**
```
/v1/basin/{data_type}/{var}/{temporal_scale}/{basinID}[.ext]
```

**Parameters:**
- `data_type`: "cru" (historical) or GCM model name (projected)
- `var`: "tas" (temperature in Celsius) or "pr" (precipitation in mm)
- `temporal_scale`: "year" or "month"
- `ISO3`: Three-letter country code (USA, KEN, etc.)
- `ext`: "json", "xml", or "csv" (default: xml)

**Example requests:**
- Temperature (monthly, Kenya): `http://climatedataapi.worldbank.org/climateweb/rest/v1/country/cru/tas/month/KEN.json`
- Precipitation (yearly, USA): `http://climatedataapi.worldbank.org/climateweb/rest/v1/country/cru/pr/year/USA.json`

### Response Format

JSON response contains array of objects with:
- `year` or `month`: Time identifier
- Value: Temperature (Celsius) or precipitation (mm)

### Sources

Research based on:
- [Climate Data API Documentation](https://datahelpdesk.worldbank.org/knowledgebase/articles/902061-climate-data-api)
- [World Bank Climate Change Knowledge Portal](https://worldbank.github.io/climateknowledgeportal/README.html)
- [How to download Climate Change data from the World Bank Data API with Python](https://hatarilabs.com/ih-en/how-to-download-climate-change-data-from-the-world-bank-data-api-with-python)
- [The World Bank Climate API (Medium)](https://praveenjayasuriya.medium.com/the-world-bank-climate-api-bb1696b6d7a7)

---
