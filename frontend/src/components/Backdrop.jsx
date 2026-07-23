/* Quiet dot-grid, faded at the edges — shared between the public landing page
   and the signed-in app shell so both read as one product. No orbs, no glass. */
export default function Backdrop({ maskY = 25 }) {
  return (
    <div
      className="fixed inset-0 pointer-events-none"
      style={{
        backgroundImage: 'radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px)',
        backgroundSize: '26px 26px',
        maskImage: `radial-gradient(ellipse 90% 70% at 50% ${maskY}%, #000 35%, transparent 100%)`,
        WebkitMaskImage: `radial-gradient(ellipse 90% 70% at 50% ${maskY}%, #000 35%, transparent 100%)`,
        zIndex: 0,
      }}
    />
  )
}
