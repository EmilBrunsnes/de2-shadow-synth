import pyo

class Synth:
    def __init__(self, out_device = None):
        self.s = pyo.Server(sr=44100, buffersize=512, nchnls=2, duplex=0)
        if out_device is None:
            out_device = pyo.pa_get_default_output()
        self.s.setOutputDevice(out_device)
        
        self.s.boot()
        if not self.s.getIsBooted():
            raise RuntimeError("Pyo server failed to boot - check device index / sample rate")
        self.s.start()
        
        self.tone = pyo.Sine(freq = 440, mul = 0.3)
        
        self.drive_sig = pyo.Sig(0.5)
        self.effect = pyo.Chorus(self.tone, depth = 0.5, feedback = 0.2)
        
        hot_signal = self.effect * self.drive_sig * 10
        
        self.disto = pyo.Disto(hot_signal, drive = 0.5, slope = 0.5).out()
        
    def update(self, freq, drive):
        self.effect.feedback = float(freq)
        self.drive_sig.value = float(drive)
        
    def stop(self):
        self.s.stop()
        self.s.shutdown()
