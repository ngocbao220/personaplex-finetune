# Third-party notices

`src/moshi` is a vendored runtime dependency from the PersonaPlex/Moshi source
tree, copied at upstream revision `3428dfd95309a7f3c84fd93259ded0f810d1ff91`.
It is included so this repository can run after a normal Git clone without a
separate PersonaPlex source checkout.

The applicable MIT license texts are retained verbatim in:

- `third_party_licenses/personaplex-MIT.txt`
- `third_party_licenses/moshi-MIT.txt`
- `third_party_licenses/audiocraft-MIT.txt`

The vendored package is used only with explicitly downloaded local checkpoint
files; the fine-tuning runtime does not call Hugging Face download helpers.
