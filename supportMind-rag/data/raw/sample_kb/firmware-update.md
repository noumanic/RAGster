# Updating firmware

The Helios X4 ships with firmware v2.1.x and is upgraded to v2.4.x or later
during initial setup. Updates after that are delivered over-the-air every
quarter.

## Automatic updates

By default, the device checks for updates every 24 hours and applies them
overnight. You can change the schedule in **Settings → Maintenance → Update
window**.

## Manual update via the app

1. Open the Helios companion app.
2. Go to **Device → Firmware**.
3. Tap **Check for updates**.
4. If an update is available, tap **Install now**.

The device reboots once during installation; expect 2–4 minutes of downtime.

## Manual update via CLI

For homelab users:

```
helios-cli firmware fetch --channel stable
helios-cli firmware apply --confirm
```

The `stable` channel lags `beta` by ~3 weeks. Beta firmware is unsigned for
some hardware revisions; check `helios-cli info --hw` before switching channels.

## Update failed — error E_FW_017

E_FW_017 means the downloaded firmware image failed signature verification.
This almost always indicates a corrupted download. Run:

```
helios-cli firmware verify
helios-cli firmware fetch --force
```

If the second fetch fails verification too, the device's TPM may have a stale
trust anchor — contact support and include the output of
`helios-cli diag --secureboot`.

## Rolling back

Rollback is supported within a single major version (e.g. 2.4.3 → 2.4.2) but
not across major versions (you cannot roll back from 3.x to 2.x). Use
`helios-cli firmware rollback --to 2.4.2`.
