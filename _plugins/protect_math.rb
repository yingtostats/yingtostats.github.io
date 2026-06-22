# Protect math spans from kramdown.
#
# kramdown processes markdown *inside* inline `$...$` math (single-dollar is not
# kramdown math syntax), so subscripts like `$\tilde{x}_{j}$` get their
# underscores turned into <em> tags and `\{ \}` collapsed — breaking MathJax.
#
# This hook lifts every math span out of the source *before* kramdown runs,
# leaving an inert placeholder, then drops the original math back in *after*
# rendering. kramdown never touches math; MathJax receives it intact.
# Source stays single-`$`; no MathJax/engine changes needed.
module ProtectMath
  # $$...$$ (may span lines), or single-line $...$ with no inner $.
  MATH = /\$\$.+?\$\$|\$[^$\n]+?\$/m
  TOKEN = ->(i) { "zZmathspanZz#{i}zZ" }

  def self.lift(content)
    store = []
    out = content.gsub(MATH) do |m|
      store << m
      TOKEN.call(store.length - 1)
    end
    [out, store]
  end

  def self.drop(output, store)
    store.each_with_index { |m, i| output = output.sub(TOKEN.call(i), m) }
    output
  end
end

Jekyll::Hooks.register [:posts, :pages], :pre_render do |doc|
  next unless doc.respond_to?(:content) && doc.content
  doc.content, store = ProtectMath.lift(doc.content)
  doc.data["__math_store"] = store
end

Jekyll::Hooks.register [:posts, :pages], :post_render do |doc|
  store = doc.data && doc.data["__math_store"]
  next unless store && doc.output
  doc.output = ProtectMath.drop(doc.output, store)
end
