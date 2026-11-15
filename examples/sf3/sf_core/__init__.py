"""The SF3 emulator interface, vendored from the predecessor project.

`moves.py` and `env.py` came from the MAMEToolkit-based harness that project
built and proved -- frame-exact macro actions over a real ROM in a Modal
Sandbox. They are vendored rather than imported because an autobranch bundle
has to be self-contained: the framework uploads the bundle and builds its
image from the manifest, so a bundle that reached into a sibling repository
would not run anywhere but this laptop.

Only the import paths changed (`sf.vendor.moves` -> `sf_core.moves`). The
emulator logic is untouched, deliberately: it encodes frame timings and memory
addresses that were established by measurement against the real ROM, and
re-deriving them would be the most expensive possible way to learn nothing new.
"""
