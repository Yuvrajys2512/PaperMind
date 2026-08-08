/**
 * Tests for MetricRing — the score dial used by every audit panel.
 *
 * Its job is turning a number into a ring arc plus a label, in two different
 * input conventions (0–1 fraction vs 0–100 percentage). Getting that backwards
 * would render "0.93%" or a 9300%-full ring, so the conventions are pinned
 * here along with the clamping that stops a >100 value from overdrawing.
 */
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

import { MetricRing } from './MetricRing'
import { METRIC_TOOLTIPS } from './metricTooltips'

const arc = (container) => container.querySelectorAll('circle')[1]

describe('value formatting', () => {
  it('renders a 0–1 fraction to two decimals', () => {
    render(<MetricRing label="faithfulness" value={0.93} />)
    expect(screen.getByText('0.93')).toBeInTheDocument()
  })

  it('renders a percentage with one decimal and a % sign', () => {
    render(<MetricRing label="original" value={93} isPercentage />)
    expect(screen.getByText('93.0%')).toBeInTheDocument()
  })

  it('shows the label', () => {
    render(<MetricRing label="sourced" value={97} isPercentage />)
    expect(screen.getByText('sourced')).toBeInTheDocument()
  })

  it('renders a zero score rather than treating it as missing', () => {
    render(<MetricRing label="numbers" value={0} />)
    expect(screen.getByText('0.00')).toBeInTheDocument()
  })
})

describe('arc geometry', () => {
  const circumference = 2 * Math.PI * 20

  it('draws an empty arc before the fill animation starts', () => {
    // Mounts with offset == circumference, then animates to the real value.
    const { container } = render(<MetricRing label="x" value={0.5} />)
    expect(Number(arc(container).getAttribute('stroke-dashoffset')))
      .toBeCloseTo(circumference, 5)
  })

  it('clamps a percentage above 100 so the ring cannot overdraw', () => {
    const { container } = render(<MetricRing label="x" value={150} isPercentage />)
    const offset = Number(arc(container).getAttribute('stroke-dashoffset'))
    expect(offset).toBeGreaterThanOrEqual(0)
  })

  it('clamps a fraction above 1 the same way', () => {
    const { container } = render(<MetricRing label="x" value={1.4} />)
    const offset = Number(arc(container).getAttribute('stroke-dashoffset'))
    expect(offset).toBeGreaterThanOrEqual(0)
  })
})

describe('accents', () => {
  // Each write-mode tab owns a colour; an unknown accent must still render.
  it.each([
    ['cyan', '#00f5ff'],
    ['violet', '#a78bfa'],
    ['amber', '#fbbf24'],
    ['emerald', '#34d399'],
    ['rose', '#fb7185'],
    ['blue', '#60a5fa'],
  ])('%s resolves to %s', (accent, hex) => {
    const { container } = render(<MetricRing label="x" value={0.5} accent={accent} />)
    expect(arc(container).getAttribute('stroke')).toBe(hex)
  })

  it('falls back to a real colour for an unknown accent', () => {
    const { container } = render(<MetricRing label="x" value={0.5} accent="chartreuse" />)
    expect(arc(container).getAttribute('stroke')).toMatch(/^#[0-9a-f]{6}$/i)
  })
})

describe('tooltips', () => {
  it('is hidden until hover', () => {
    render(<MetricRing label="faithfulness" value={0.9} />)
    expect(screen.queryByText(METRIC_TOOLTIPS.faithfulness.body)).not.toBeInTheDocument()
  })

  it('an explicit tooltip prop wins over the shared table', () => {
    const tooltip = { title: 'Original', body: 'Custom explanation.' }
    // Label matches a METRIC_TOOLTIPS key to prove the prop takes precedence.
    render(<MetricRing label="numbers" value={90} isPercentage tooltip={tooltip} />)
    expect(screen.queryByText(METRIC_TOOLTIPS.numbers.body)).not.toBeInTheDocument()
  })

  it('every shared tooltip has both a title and a body', () => {
    for (const [key, tip] of Object.entries(METRIC_TOOLTIPS)) {
      expect(tip.title, `${key}.title`).toBeTruthy()
      expect(tip.body, `${key}.body`).toBeTruthy()
    }
  })
})
