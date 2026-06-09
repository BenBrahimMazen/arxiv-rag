# Recording the demo GIF

The README embeds `docs/demo.gif`. Record it once the app is running locally
(`docker compose up` + `python -m scripts.ingest --max-papers 50`).

## What to capture (~10–15s)

1. Open http://localhost:8501
2. Click an example question (or type one)
3. Show the answer **streaming in** token by token
4. Expand the **Sources** section to reveal the paper cards with `arxiv:` ids
5. Stop the recording

## Tools

| OS | Tool |
| --- | --- |
| Windows | [ScreenToGif](https://www.screentogif.com/) (free, records straight to GIF) |
| macOS | [Kap](https://getkap.co/) → export GIF |
| Linux | [Peek](https://github.com/phw/peek) |
| Any | Record MP4, then convert (below) |

## Convert MP4 → optimized GIF (ffmpeg)

```bash
ffmpeg -i demo.mp4 -vf "fps=12,scale=900:-1:flags=lanczos,palettegen" palette.png
ffmpeg -i demo.mp4 -i palette.png -vf "fps=12,scale=900:-1:flags=lanczos,paletteuse" docs/demo.gif
```

Keep it under ~5 MB so it loads fast on GitHub. Then commit `docs/demo.gif`.
