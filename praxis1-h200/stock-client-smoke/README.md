# Praxis DROID stock Positronic client smoke

On one H200, the stock Positronic `RemotePolicy` at `62b6fbd54e1786c938ec3e457581ca902bb3b2aa` connected to a disposable serving-gate copy of M* #1520 at `5902054c0f9fa057a80ae5a7e15e67f19d4bcde2`. The source-matched Rust server ran protocol v2. This was one synthetic, 27-tick open-loop session.

The client applied the advertised local stack: both raw 288x512 camera views became 234x416 JPEG observations, with 1 frame on the cold call and 22 on the warm call. The same session returned 21 then 24 typed command rows with `robot_command` and `target_grip`, and closed once.

[Report](report.json) | [events](events.jsonl) | [close](close.json) | [source hashes](source-files.sha256) | [served config hash](mstar-served.sha256)

The probe checked command types, keys and row counts, but did not record or validate the numeric contents. It did not run a closed-loop simulation or measure performance. The committed serving gate stayed off.
