# Device won't charge

If your Helios X4 device shows no charging indicator when plugged in, try the
steps below in order. Most charging problems resolve at step 2 or 3.

## Step 1: Check the cable

Use only the official USB-C PD 3.0 cable rated for at least 65W. Third-party
cables that only support USB 2.0 data may not negotiate Power Delivery
correctly with the Helios X4 charger and will trigger error code E_CHG_002
(handshake failed).

## Step 2: Hard reset the charging controller

1. Unplug the charger.
2. Hold the power button for 15 seconds.
3. Wait 60 seconds.
4. Plug the charger back in.

The amber LED should blink twice within 5 seconds. If it does not, proceed to
step 3.

## Step 3: Inspect the USB-C port

Lint and dust commonly cause intermittent charging. Use a wooden toothpick — never
metal — to gently clear the port. If you see green corrosion on the contacts,
the device must be serviced; do not attempt to clean it with liquids.

## Step 4: Try a different outlet

Some power strips with surge protection downgrade USB-C PD negotiations. Plug
the charger directly into a wall outlet. If charging resumes, the issue is with
the power strip, not the device.

## When to contact support

If none of the steps above resolve the problem, contact support with the
output of `helios-cli diag --power` and the serial number printed on the back
of the device.
