import Link from "next/link";

export default function GuidePage() {
  return (
    <div className="panel guide">
      <h1 className="page-title">Guide</h1>
      <p>
        AeroDoppler proposes where a recording was made by matching the pitch changes of propeller aircraft in the audio (Doppler <i>ridges</i>) against recorded flight tracks (ADS-B). It
        runs locally; nothing is uploaded. Analyse only recordings you own or may analyse. The full user guide is <code>docs/USER_GUIDE.md</code> in the repository.
      </p>

      <h2>Try it first</h2>
      <p>
        <code>make demo</code> builds a SYNTHETIC recording, flights and terrain and runs both search modes (about 3 minutes). Open the results it prints to see what a good answer looks like.
      </p>

      <h2>The workflow</h2>
      <ol>
        <li>
          <b>Import a recording</b> (<Link href="/">Import</Link>) and say what you know about <b>when</b> it was recorded: an exact UTC start (with its uncertainty), only the UTC day, a range of
          days, only an upper bound, or nothing. A video&apos;s publication date is only an upper bound — the recording can be older. Files longer than 2 hours are analysed in 2-hour
          segments; all times stay tied to the original file.
        </li>
        <li>
          <b>Import the flight archive of that UTC day.</b> ADSB.lol publishes one whole-world archive per day (≈ 3.5 GB). It is downloaded once and cached even for a small area. Days appear a
          few hours after they end; archives exist from February 2023. The import runs in the background — follow it on the <Link href="/jobs/">Jobs</Link> page; reloading the browser does not
          stop it.
        </li>
        <li>
          <b>Mark aircraft passes</b> in the <Link href="/lab/">Spectrogram lab</Link>: automatic detection scans the whole recording and explains what it rejected; you can also draw a ridge
          by hand. Each event gets an evidence label: <span className="tag ev-good">plausible aircraft Doppler</span> (a descending pitch transition of several percent),{" "}
          <span className="tag ev-weak">weak Doppler evidence</span>, or <span className="tag ev-bad">tonal ridge only</span> (no measurable transition — hum, machinery, music, or an
          aircraft heard only on a plateau; it cannot locate anything).
        </li>
        <li>
          <b>Search</b> (<Link href="/search/">Search</Link>). <i>Unknown clock time</i> needs the UTC day and two passes: their time separation replaces the clock. <i>Known start time</i>{" "}
          needs the verified UTC start and its uncertainty; one pass can be enough but never reaches a high grade. The page lists what is missing before it lets you run.
        </li>
      </ol>

      <h2>Reading a result</h2>
      <ul>
        <li>
          Scores are <b>uncalibrated relative fit values</b>; differences between candidates matter, absolute values do not. Nothing is a probability or a confidence interval.
        </li>
        <li>
          The <b>grade</b> (high / medium / low) is a rule-based verdict from named checks shown with the result. <b>LOW means the coordinates are not a verified location.</b> Even HIGH is
          not a guarantee: on single-pass real recordings, high answers have still been hundreds of metres off.
        </li>
        <li>
          The &ldquo;score plateau&rdquo; radius is how far the score stays near its maximum — a sign of how well-pinned the answer is, not an error bar.
        </li>
        <li>
          A search area you chose from prior knowledge (a title, a guess) makes the result a <b>bounded</b> search: the answer could only be found inside that area.
        </li>
      </ul>

      <h2>When something goes wrong</h2>
      <ul>
        <li>
          <b>No events detected</b>: open &ldquo;rejected candidates&rdquo; under the detection summary; lower the minimum duration or threshold; look for slanted lines yourself and draw them.
        </li>
        <li>
          <b>Archive not available</b>: the day has not been published yet (retry a few hours after the UTC day ends), predates February 2023, or was skipped by the provider. No other day is
          ever substituted.
        </li>
        <li>
          <b>A job stopped</b>: the Jobs page shows it as interrupted or failed with the reason. Retrying an import resumes the download and reuses cached archive parts.
        </li>
        <li>
          <b>Memory</b>: imports and searches size their worker count from the memory actually available (container/WSL limits included) and stop with an explanation instead of exhausting
          RAM. On WSL2 the Windows-side <code>.wslconfig</code> <code>memory=</code> setting caps what Linux can use.
        </li>
      </ul>
    </div>
  );
}
