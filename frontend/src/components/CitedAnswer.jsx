const ACCENT = '#22d3ee'

/* The product proof, de-glassed. The old version used backdrop-blur + a white
   gradient (the exact glassmorphism look we're moving away from). This is a
   solid, high-contrast panel with crisp hairlines — an editorial "screenshot"
   of the actual mechanic: an answer whose number links to the exact source line
   it was drawn from. Reused on both the public landing and the signed-in home. */
export default function CitedAnswer() {
  return (
    <div
      className="rounded-2xl w-full overflow-hidden"
      style={{ background: '#0d0f14', border: '1px solid rgba(255,255,255,0.09)' }}
    >
      {/* window chrome */}
      <div
        className="flex items-center gap-1.5 px-4 py-3 min-w-0"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}
      >
        <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: 'rgba(255,255,255,0.14)' }} aria-hidden="true" />
        <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: 'rgba(255,255,255,0.14)' }} aria-hidden="true" />
        <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: 'rgba(255,255,255,0.14)' }} aria-hidden="true" />
        <span className="ml-2 text-[10px] text-gray-500 truncate min-w-0" style={{ fontFamily: 'var(--font-mono)' }}>
          attention-is-all-you-need.pdf
        </span>
      </div>

      <div className="p-5">
        {/* question */}
        <div className="flex justify-end mb-4">
          <div
            className="text-[13px] text-gray-100 px-3.5 py-2 rounded-2xl rounded-br-md max-w-[80%]"
            style={{ background: 'rgba(255,255,255,0.07)' }}
          >
            What was the main result?
          </div>
        </div>

        {/* answer */}
        <div className="text-[13px] text-gray-200 leading-relaxed mb-4">
          The Transformer reached <span className="text-white font-semibold">28.4 BLEU</span> on
          WMT14 EN→DE, improving over the best prior result by more than 2 BLEU
          <span
            className="align-super text-[10px] font-semibold px-1.5 py-0.5 rounded-md mx-0.5"
            style={{ background: `${ACCENT}1f`, color: ACCENT, fontFamily: 'var(--font-mono)' }}
          >
            §6.2
          </span>
          while training in a fraction of the time.
        </div>

        {/* the exact source line the number came from — the whole point */}
        <div
          className="rounded-xl px-3.5 py-3 mb-4"
          style={{ background: 'rgba(34,211,238,0.05)', border: `1px solid ${ACCENT}2e` }}
        >
          <div className="text-[9px] uppercase tracking-[0.16em] text-gray-500 mb-1.5" style={{ fontFamily: 'var(--font-mono)' }}>
            §6.2 · Results
          </div>
          <p className="text-[12px] text-gray-300 leading-relaxed">
            “…our model achieves <span className="text-white font-semibold" style={{ background: `${ACCENT}22`, padding: '0 3px', borderRadius: 3 }}>28.4 BLEU</span> on the WMT 2014 English-to-German translation task, outperforming the best previously reported models by over 2 BLEU.”
          </p>
        </div>

        {/* grounded footer */}
        <div className="flex items-center gap-3 pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
          <span className="flex items-center gap-1.5 text-[11px] text-gray-400" style={{ fontFamily: 'var(--font-mono)' }}>
            <span className="w-1.5 h-1.5 rounded-full ed-pulse" style={{ background: ACCENT, boxShadow: `0 0 6px ${ACCENT}` }} />
            grounded in source
          </span>
          <span className="text-[11px] text-gray-600" style={{ fontFamily: 'var(--font-mono)' }}>confidence 94%</span>
        </div>
      </div>
    </div>
  )
}
