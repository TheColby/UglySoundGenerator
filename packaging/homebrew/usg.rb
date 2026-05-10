class Usg < Formula
  desc "Ugly Sound Generator for psychoacoustic, chiptune, and multichannel mayhem"
  homepage "https://github.com/TheColby/UglySoundGenerator"
  head "https://github.com/TheColby/UglySoundGenerator.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", "--locked", "--path", ".", "--root", prefix
  end

  test do
    assert_match "UglySoundGenerator", shell_output("#{bin}/usg --help")
    system "#{bin}/usg", "render", "--duration", "0.1", "--style", "hum", "--output", testpath/"hum.wav"
    assert_predicate testpath/"hum.wav", :exist?
  end
end
