/* Shared PaperMind wordmark — the logo mark + name, used in both the public
   landing nav and the signed-in home nav so they read as one product. */
export default function Wordmark({ accent = '#22d3ee' }) {
  return (
    <div className="flex items-center gap-2.5 select-none">
      <div
        className="w-7 h-7 rounded-lg flex items-center justify-center"
        style={{ background: `${accent}1a`, border: `1px solid ${accent}33` }}
      >
        <svg className="w-4 h-4" fill="none" stroke={accent} viewBox="0 0 24 24">
          <path
            strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
            d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
          />
        </svg>
      </div>
      <span
        className="text-[15px] font-semibold text-white tracking-tight"
        style={{ fontFamily: 'var(--font-display)' }}
      >
        PaperMind
      </span>
    </div>
  )
}
