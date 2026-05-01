# Wi-Fi setup and troubleshooting

The Helios X4 supports both 2.4GHz and 5GHz networks (802.11 a/b/g/n/ac).
Most connection problems fall into one of three categories: SSID broadcast,
band steering, or DHCP lease.

## First-time setup

1. Open the Helios companion app.
2. Tap **Add device** and choose **Helios X4**.
3. Scan the QR code on the bottom of the device.
4. Enter your Wi-Fi password when prompted.

The setup uses BLE to push credentials, so the device does not need to be on
the same network as your phone yet.

## Error code W_NET_004 — "could not join network"

This error means the device received the SSID but could not authenticate. Most
common causes:

- WPA3-only routers. The Helios X4 supports WPA2-PSK and WPA3 transition mode,
  but does not support WPA3 personal alone. Switch the router to WPA2/WPA3
  transition.
- Hidden SSIDs. The Helios X4 cannot join a hidden network during onboarding.
  Temporarily un-hide the SSID, complete setup, and re-hide it afterwards.
- 5GHz-only mesh networks. If your mesh runs band steering, force the
  onboarding device onto the 2.4GHz band for setup.

## Slow downloads after a successful connection

If `helios-cli diag --net` reports RSSI worse than -75dBm, the device is too
far from the access point. Move it closer or add a mesh node.

## Resetting Wi-Fi credentials

Hold the **Mode** and **Power** buttons together for 10 seconds until the LED
flashes red three times. This wipes saved Wi-Fi credentials only — it does not
factory-reset the device.
