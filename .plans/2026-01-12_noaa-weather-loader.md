# Plan: NOAA Weather Loader

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a loader for NOAA Climate Data Online (CDO) API to fetch historical weather station data with proper dimensional units (temperature, precipitation, wind speed, pressure).

---

## Background

NOAA provides extensive historical weather data via their API. The existing NOAAClimateLoader in climate.py is a placeholder that raises NotImplementedError. This plan will implement full functionality for loading weather station data from NOAA's NCEI (National Centers for Environmental Information) API.

Key resources:
- NOAA CDO Web Services API v2: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
- Token request: https://www.ncdc.noaa.gov/cdo-web/token
- New NCEI Data Service API v1: https://www.ncei.noaa.gov/access/services/data/v1
- API rate limits: 5 requests/second, 10,000 requests/day per token

Note: The v2 API is deprecated but still functional. For future-proofing, we should support both v2 (for now) and document migration path to v1.

---

## Approach

### Option A: V2 API with token authentication
- Description: Use existing CDO v2 API with email-based token authentication
- Pros:
  - Well-documented endpoint structure
  - Simple JSON responses
  - Station search by location/date range
  - Direct access to GHCND (daily summaries) dataset
- Cons:
  - Deprecated (though still functional)
  - Rate limits require careful handling
  - Requires user to obtain API token separately

### Option B: V1 API with different authentication
- Description: Use newer NCEI Data Service API v1
- Pros:
  - Future-proof solution
  - Modern API design
- Cons:
  - Less documentation available
  - Different data access patterns
  - May require different authentication

### Option C: Hybrid with sample data fallback
- Description: Implement v2 API but provide synthetic sample data as fallback
- Pros:
  - Works without API token for demos
  - Production-ready when token provided
  - Follows pattern from PRISMClimateLoader
  - Easy to test
- Cons:
  - More code complexity
  - Sample data doesn't represent real locations

### Decision: Option C (Hybrid approach)

Best of both worlds - implement full v2 API support but allow demo usage without token. This matches the existing pattern in PRISMClimateLoader and makes testing easier. We'll document the v2 deprecation and provide clear upgrade path in comments.

---

## Implementation Steps

1. [ ] Update NOAAClimateLoader class in climate.py:
   - Add API_URL constant for v2 endpoint
   - Add token parameter to __init__
   - Implement token validation and warning messages

2. [ ] Implement station search functionality:
   - Add method _search_stations(location, start_date, end_date)
   - Support location by lat/lon or location ID
   - Cache station results

3. [ ] Implement data fetching:
   - Add method _fetch_weather_data(station_id, dataset_id, start_date, end_date, datatypes)
   - Support GHCND dataset (daily summaries)
   - Handle pagination for large date ranges
   - Implement rate limiting (5 req/sec)

4. [ ] Add weather variable parsing:
   - Parse TMAX, TMIN, TAVG (temperature in tenths of °C)
   - Parse PRCP (precipitation in tenths of mm)
   - Parse SNOW (snowfall in mm)
   - Parse AWND (wind speed in tenths of m/s)
   - Parse PRES (pressure in Pa or hPa)

5. [ ] Create units if needed:
   - Check if m/s (velocity) unit exists, if not create it
   - Use existing: celsius, kelvin, mm, meter, pascal from core.units
   - Add hectopascal (hPa) if needed for pressure

6. [ ] Implement DimArray conversion:
   - Convert API JSON response to numpy arrays
   - Apply scale factors (tenths to actual values)
   - Create DimArray with appropriate units
   - Handle missing data (NaN values)

7. [ ] Add sample data fallback:
   - Create _create_sample_weather_data() method
   - Generate realistic synthetic weather patterns
   - Support multiple variables matching API structure

8. [ ] Update load() method signature:
   - Parameters: location, start_date, end_date, variables, station_id
   - Return dict with metadata + DimArrays
   - Fallback to sample data if no token provided

9. [ ] Add error handling:
   - Handle API rate limits (429 status)
   - Handle invalid tokens (401 status)
   - Handle network failures
   - Provide helpful error messages

10. [ ] Update __init__.py exports

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/datasets/loaders/climate.py | Replace NOAAClimateLoader NotImplementedError with full implementation |
| src/dimtensor/core/units.py | Add m_per_s (meter/second) if not exists, add hectopascal |
| tests/test_dataset_loaders.py | Add TestNOAAClimateLoader test class with mock API tests |
| src/dimtensor/datasets/loaders/__init__.py | Update imports/exports (likely already correct) |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit tests for token validation and warnings
- [ ] Unit tests for _search_stations with mocked API responses
- [ ] Unit tests for _fetch_weather_data with mocked pagination
- [ ] Unit tests for weather variable parsing and unit conversion
- [ ] Unit tests for sample data fallback (no token scenario)
- [ ] Integration test for complete load() workflow with mocked API
- [ ] Test rate limiting behavior (mocked)
- [ ] Test error handling (401, 429, network errors)
- [ ] Manual verification with real API token (optional, marked with @pytest.mark.network)

---

## Risks / Edge Cases

- **Risk**: API v2 deprecation leads to shutdown
  - **Mitigation**: Document clearly in code comments, add warning log when using v2, provide migration guide to v1 in docstring

- **Risk**: Rate limits cause failures in tests or production
  - **Mitigation**: Implement exponential backoff, cache aggressively, document rate limits in docstring

- **Risk**: User doesn't have API token
  - **Mitigation**: Provide sample data fallback, clear instructions in error message about obtaining token

- **Edge case**: Large date ranges cause pagination
  - **Handling**: Implement pagination loop in _fetch_weather_data, limit default to 1 year of data

- **Edge case**: Station has missing data for requested variables
  - **Handling**: Return NaN arrays, document in return value dict which variables are available

- **Edge case**: Temperature offset conversion (celsius vs kelvin)
  - **Handling**: NOAA provides celsius, store as celsius unit (relative temp), document that absolute temp requires kelvin + offset

- **Edge case**: Pressure units (Pa vs hPa)
  - **Handling**: NOAA typically uses hPa, convert to Pa (SI) or keep hPa if that's more common in climate science

- **Edge case**: Wind speed averaging periods
  - **Handling**: Document which wind variable we're using (AWND = average daily wind speed)

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass (including mocked API tests)
- [ ] Sample data fallback works without token
- [ ] Real API integration works with token (manual test)
- [ ] Documentation/docstrings updated with:
  - How to obtain API token
  - Rate limit warnings
  - V2 deprecation notice
  - Example usage
- [ ] Code follows patterns from NISTCODATALoader and PRISMClimateLoader
- [ ] Units properly defined (m_per_s, hPa if needed)
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**Key Design Decisions:**

1. **Units to add/use:**
   - Temperature: `celsius`, `kelvin` (already exist in core.units)
   - Precipitation: `mm` (already exists)
   - Wind speed: `meter / second` (construct dynamically) or add `m_per_s` constant
   - Pressure: `pascal` (exists), add `hectopascal = 100 * pascal`

2. **NOAA API data types (GHCND):**
   - TMAX, TMIN, TAVG: Temperature in tenths of °C → divide by 10
   - PRCP: Precipitation in tenths of mm → divide by 10
   - SNOW: Snowfall in mm → use directly
   - AWND: Average wind speed in tenths of m/s → divide by 10
   - Note: Not all stations report all variables

3. **Return structure:**
   ```python
   {
       'station_id': str,
       'station_name': str,
       'latitude': float,
       'longitude': float,
       'elevation': DimArray (meter),
       'dates': list[str],  # ISO format dates
       'temperature_max': DimArray (celsius),
       'temperature_min': DimArray (celsius),
       'precipitation': DimArray (mm),
       'wind_speed': DimArray (m/s),
       'pressure': DimArray (hPa or Pa),
       # ... other variables as available
   }
   ```

4. **Token handling:**
   - Accept token via parameter: `NOAAClimateLoader(token="abc123")`
   - Also check environment variable: `NOAA_API_TOKEN`
   - If no token, use sample data and log warning

5. **Caching strategy:**
   - Use BaseLoader.download() for API responses
   - Cache key: hash of (station_id, start_date, end_date, datatypes)
   - Respect cache_enabled setting

---
