// App.tsx — Main chat UI for the Medical RAG booking system.
// Handles: conversational intake, provider match display, time slot selection,
// mock payment flow, booking confirmation, and Email/WhatsApp/Telegram sharing.

import { Mic, Send, Stethoscope, User, Bot } from "lucide-react";
import { useEffect, useRef, useState } from "react";

type ChatResponse = {
  session_id: string;
  reply: string;
  status?: "needs_info" | "emergency" | "matched" | "booked";
  candidates?: ProfessionalCandidate[];
};

type ProfessionalCandidate = {
  id: string;
  name: string;
  specialty: string;
  city: string;
  score?: number;
  address?: string | null;
};

type AvailabilityResponse = {
  professional_id: string;
  slots: string[];
};

type ChatMessage = {
  role: "user" | "agent";
  text: string;
};

type PaymentContext = {
  doctor: ProfessionalCandidate;
  slot: string;
};

type BookingReceipt = {
  doctorName: string;
  specialty: string;
  city: string;
  slot: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<ProfessionalCandidate[]>([]);
  const [slotsByDoctor, setSlotsByDoctor] = useState<Record<string, string[]>>({});
  const [selectedSlotByDoctor, setSelectedSlotByDoctor] = useState<Record<string, string>>({});
  const [bookedSlots, setBookedSlots] = useState<Record<string, boolean>>({});
  const [paymentContext, setPaymentContext] = useState<PaymentContext | null>(null);
  const [payName, setPayName] = useState("");
  const [payCard, setPayCard] = useState("");
  const [payExpiry, setPayExpiry] = useState("");
  const [payCvv, setPayCvv] = useState("");
  const [isPaying, setIsPaying] = useState(false);
  const [paymentError, setPaymentError] = useState<string | null>(null);
  const [receipt, setReceipt] = useState<BookingReceipt | null>(null);
  const [contactEmail, setContactEmail] = useState("");
  const [contactPhone, setContactPhone] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [recorder, setRecorder] = useState<MediaRecorder | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMsg = input.trim();
    setMessages((prev) => [...prev, { role: "user", text: userMsg }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg, session_id: sessionId }),
      });

      if (!res.ok) {
        setMessages((prev) => [
          ...prev,
          { role: "agent", text: "I could not process your request. Please try again." },
        ]);
        return;
      }

      const data = (await res.json()) as ChatResponse;
      setSessionId(data.session_id);
      setMessages((prev) => [...prev, { role: "agent", text: data.reply }]);
      if (data.status === "matched" && data.candidates?.length) {
        setCandidates(data.candidates);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "agent", text: "Connection error. Please check if backend is running." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleVoice = async (file: File) => {
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      if (sessionId) form.append("session_id", sessionId);
      const res = await fetch(`${API_BASE}/api/voice`, { method: "POST", body: form });

      if (!res.ok) {
        setMessages((prev) => [
          ...prev,
          { role: "agent", text: "Voice processing failed. Please try again." },
        ]);
        return;
      }

      const data = (await res.json()) as ChatResponse;
      setSessionId(data.session_id);
      setMessages((prev) => [...prev, { role: "agent", text: data.reply }]);
      if (data.status === "matched" && data.candidates?.length) {
        setCandidates(data.candidates);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "agent", text: "Voice connection error. Please check backend status." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        void handleVoice(file);
        stream.getTracks().forEach((track) => track.stop());
      };
      mediaRecorder.start();
      setRecorder(mediaRecorder);
      setIsRecording(true);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "agent", text: "Microphone access denied. Please enable microphone permission." },
      ]);
    }
  };

  const stopRecording = () => {
    if (!recorder) return;
    recorder.stop();
    setRecorder(null);
    setIsRecording(false);
  };

  const loadSlots = async (doctorId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/availability?professional_id=${encodeURIComponent(doctorId)}`);
      if (!res.ok) return;
      const data = (await res.json()) as AvailabilityResponse;
      setSlotsByDoctor((prev) => ({ ...prev, [doctorId]: data.slots || [] }));
    } catch {
      // no-op
    }
  };

  const confirmMockBooking = (doctor: ProfessionalCandidate) => {
    const slot = selectedSlotByDoctor[doctor.id];
    if (!slot) return;
    setPaymentError(null);
    setPaymentContext({ doctor, slot });
  };

  const completeMockPayment = async () => {
    if (!paymentContext) return;
    if (!payName.trim() || !payCard.trim() || !payExpiry.trim() || !payCvv.trim()) {
      setPaymentError("Please complete all payment fields.");
      return;
    }
    setPaymentError(null);
    setIsPaying(true);

    try {
      await fetch(`${API_BASE}/api/bookings/mock`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId || "",
          professional_name: paymentContext.doctor.name,
          specialty: paymentContext.doctor.specialty,
          city: paymentContext.doctor.city,
          appointment_date: paymentContext.slot,
        }),
      });
    } catch {
      // Booking save failed silently — still show confirmation to user
    }

    const when = new Date(paymentContext.slot).toLocaleString();
    const bookedKey = `${paymentContext.doctor.id}:${paymentContext.slot}`;
    setBookedSlots((prev) => ({ ...prev, [bookedKey]: true }));
    setReceipt({
      doctorName: paymentContext.doctor.name,
      specialty: paymentContext.doctor.specialty,
      city: paymentContext.doctor.city,
      slot: paymentContext.slot,
    });
    setMessages((prev) => [
      ...prev,
      {
        role: "agent",
        text: `Payment completed. Your slot is reserved with ${paymentContext.doctor.name} on ${when}. `
          + "Share your email or phone number so we can send your booking confirmation.",
      },
    ]);
    setCandidates([]);
    setPaymentContext(null);
    setPayName("");
    setPayCard("");
    setPayExpiry("");
    setPayCvv("");
    setPaymentError(null);
    setIsPaying(false);
  };

  const getReceiptText = () => {
    if (!receipt) return "";
    return `Booking confirmed: ${receipt.doctorName} (${receipt.specialty}) in ${receipt.city} on ${new Date(
      receipt.slot,
    ).toLocaleString()}.`;
  };

  const sendEmailConfirmation = () => {
    if (!contactEmail.trim() || !receipt) return;
    const subject = encodeURIComponent("Medical Booking Confirmation");
    const body = encodeURIComponent(getReceiptText());
    window.open(`mailto:${contactEmail.trim()}?subject=${subject}&body=${body}`, "_blank");
  };

  const sendWhatsappConfirmation = () => {
    if (!contactPhone.trim() || !receipt) return;
    const phone = contactPhone.replace(/[^\d+]/g, "");
    const text = encodeURIComponent(getReceiptText());
    window.open(`https://wa.me/${phone.replace("+", "")}?text=${text}`, "_blank");
  };

  const sendTelegramConfirmation = () => {
    if (!contactPhone.trim() || !receipt) return;
    const text = encodeURIComponent(getReceiptText());
    // Telegram deep-link by phone isn't universal; open share intent with prefilled text.
    window.open(`https://t.me/share/url?url=${encodeURIComponent("https://your-clinic.example")}&text=${text}`, "_blank");
  };

  return (
    <div
      className="min-h-screen w-screen bg-slate-100 flex items-center justify-center p-4"
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        width: "100vw",
        height: "100vh",
        backgroundColor: "#f1f5f9",
        margin: 0,
        padding: 0,
        position: "fixed",
        top: 0,
        left: 0,
      }}
    >
      <div
        className="h-[80vh] w-full max-w-2xl flex flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl"
        style={{
          width: "100%",
          maxWidth: "600px",
          height: "85vh",
          display: "flex",
          flexDirection: "column",
          backgroundColor: "white",
          borderRadius: "24px",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.25)",
          overflow: "hidden",
        }}
      >
        <header className="flex items-center gap-2 bg-blue-600 px-4 py-2.5 text-white shadow-md">
          <div className="rounded-lg bg-white/20 p-1.5">
            <Stethoscope size={18} />
          </div>
          <div>
            <h1 className="text-sm font-semibold leading-tight">AI Medical Concierge</h1>
            <p className="text-[11px] italic text-blue-100">Secure Medical Intake • Italy</p>
          </div>
        </header>

        <main className="space-y-4 bg-slate-50/50 p-4" style={{ flex: 1, overflowY: "auto" }}>
          {messages.length === 0 && (
            <div className="space-y-3 py-10 text-center">
              <div className="mb-2 inline-block rounded-full bg-blue-100 p-4 text-blue-600">
                <Bot size={40} />
              </div>
              <h2 className="text-xl font-semibold text-slate-700">How can I help you today?</h2>
              <p className="mx-auto max-w-xs text-sm text-slate-500">
                Tell me your symptoms, age, and location in Italy to find the right specialist.
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={`${msg.role}-${i}`} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`flex max-w-[85%] gap-2 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                <div
                  className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full p-2 ${
                    msg.role === "user"
                      ? "bg-medical-600 text-white"
                      : "border border-slate-200 bg-white text-slate-500 shadow-sm"
                  }`}
                >
                  {msg.role === "user" ? <User size={16} /> : <Stethoscope size={16} />}
                </div>
                <div
                  className={`rounded-2xl px-4 py-2 text-sm leading-relaxed shadow-sm ${
                    msg.role === "user"
                      ? "rounded-tr-none bg-medical-600 text-white"
                      : "rounded-tl-none border border-slate-100 bg-white text-slate-700"
                  }`}
                >
                  {msg.text}
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="flex gap-1 rounded-2xl rounded-tl-none border border-slate-100 bg-white px-4 py-2">
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-300" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-300 [animation-delay:0.2s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-300 [animation-delay:0.4s]" />
              </div>
            </div>
          )}

          {receipt && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-emerald-700">Booking Confirmed</div>
              <div className="mt-1 text-sm text-emerald-900">
                {receipt.doctorName} • {receipt.specialty} • {receipt.city}
              </div>
              <div className="text-xs text-emerald-800">
                Slot: {new Date(receipt.slot).toLocaleString()}
              </div>
              <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
                <input
                  value={contactEmail}
                  onChange={(e) => setContactEmail(e.target.value)}
                  placeholder="Email for confirmation"
                  className="w-full rounded-lg border border-emerald-200 bg-white px-3 py-2 text-sm"
                />
                <input
                  value={contactPhone}
                  onChange={(e) => setContactPhone(e.target.value)}
                  placeholder="Phone (for WhatsApp/Telegram)"
                  className="w-full rounded-lg border border-emerald-200 bg-white px-3 py-2 text-sm"
                />
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={sendEmailConfirmation}
                  disabled={!contactEmail.trim()}
                  className={`rounded-lg px-2.5 py-1.5 text-xs ${
                    contactEmail.trim() ? "bg-blue-600 text-white hover:bg-blue-700" : "bg-slate-200 text-slate-500"
                  }`}
                >
                  Send Email
                </button>
                <button
                  type="button"
                  onClick={sendWhatsappConfirmation}
                  disabled={!contactPhone.trim()}
                  className={`rounded-lg px-2.5 py-1.5 text-xs ${
                    contactPhone.trim() ? "bg-emerald-600 text-white hover:bg-emerald-700" : "bg-slate-200 text-slate-500"
                  }`}
                >
                  Send WhatsApp
                </button>
                <button
                  type="button"
                  onClick={sendTelegramConfirmation}
                  disabled={!contactPhone.trim()}
                  className={`rounded-lg px-2.5 py-1.5 text-xs ${
                    contactPhone.trim() ? "bg-sky-600 text-white hover:bg-sky-700" : "bg-slate-200 text-slate-500"
                  }`}
                >
                  Send Telegram
                </button>
              </div>
            </div>
          )}

          {candidates.length > 0 && (
            <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Best matches</div>
              {candidates.map((c) => (
                <div key={c.id} className="rounded-lg border border-slate-100 p-3">
                  <div className="text-sm font-semibold text-slate-800">{c.name}</div>
                  <div className="text-xs text-slate-500">
                    {c.specialty} • {c.city}
                  </div>
                  <div className="mt-2 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void loadSlots(c.id)}
                      className="rounded-lg bg-medical-600 px-2.5 py-1.5 text-xs text-white hover:bg-medical-700"
                    >
                      Show free slots
                    </button>
                  </div>
                  {(slotsByDoctor[c.id] || []).length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {slotsByDoctor[c.id].slice(0, 4).map((s) => (
                        <button
                          key={s}
                          type="button"
                          onClick={() => setSelectedSlotByDoctor((prev) => ({ ...prev, [c.id]: s }))}
                          className={`rounded-full px-2 py-1 text-[11px] ${
                            selectedSlotByDoctor[c.id] === s
                              ? "bg-medical-600 text-white"
                              : "bg-slate-100 text-slate-600"
                          }`}
                        >
                          {new Date(s).toLocaleString()}
                        </button>
                      ))}
                    </div>
                  )}
                  {(slotsByDoctor[c.id] || []).length > 0 && (
                    <div className="mt-2">
                      <button
                        type="button"
                        onClick={() => confirmMockBooking(c)}
                        disabled={
                          !selectedSlotByDoctor[c.id] ||
                          !!bookedSlots[`${c.id}:${selectedSlotByDoctor[c.id]}`]
                        }
                        className={`rounded-lg px-2.5 py-1.5 text-xs ${
                          selectedSlotByDoctor[c.id] &&
                          !bookedSlots[`${c.id}:${selectedSlotByDoctor[c.id]}`]
                            ? "bg-emerald-600 text-white hover:bg-emerald-700"
                            : "bg-slate-200 text-slate-500"
                        }`}
                      >
                        {bookedSlots[`${c.id}:${selectedSlotByDoctor[c.id]}`]
                          ? "Booked"
                          : "Go to payment"}
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          <div ref={scrollRef} />
        </main>

        <footer className="border-t border-slate-100 bg-white p-4">
          <div className="relative flex items-center rounded-2xl bg-slate-100 px-4 py-2 transition-all focus-within:ring-2 ring-medical-600/20">
            <button
              type="button"
              onClick={() => (isRecording ? stopRecording() : void startRecording())}
              className={`p-1 transition-colors ${
                isRecording ? "text-rose-600 hover:text-rose-700" : "text-slate-400 hover:text-medical-600"
              }`}
            >
              <Mic size={20} />
            </button>
            <input
              className="flex-1 border-none bg-transparent px-3 py-2 text-sm text-slate-700 placeholder:text-slate-400 focus:ring-0"
              placeholder="Describe your medical need..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void handleSend()}
            />
            <button
              type="button"
              onClick={() => void handleSend()}
              disabled={!input.trim() || loading}
              className={`rounded-xl p-2 transition-all ${
                input.trim() && !loading
                  ? "bg-medical-600 text-white shadow-md hover:bg-medical-700"
                  : "text-slate-300"
              }`}
            >
              <Send size={18} />
            </button>
          </div>
        </footer>
      </div>
      {paymentContext && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          onClick={() => {
            setPaymentContext(null);
            setPaymentError(null);
          }}
        >
          <div
            className="w-full max-w-md rounded-2xl bg-white p-4 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-base font-semibold text-slate-800">Mock Payment</h3>
            <p className="mt-1 text-xs text-slate-500">
              {paymentContext.doctor.name} • {new Date(paymentContext.slot).toLocaleString()}
            </p>
            <div className="mt-3 space-y-2">
              <input
                value={payName}
                onChange={(e) => setPayName(e.target.value)}
                placeholder="Cardholder name"
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
              />
              <input
                value={payCard}
                onChange={(e) => setPayCard(e.target.value)}
                placeholder="Card number (mock)"
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
              />
              <div className="grid grid-cols-2 gap-2">
                <input
                  value={payExpiry}
                  onChange={(e) => setPayExpiry(e.target.value)}
                  placeholder="MM/YY"
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                />
                <input
                  value={payCvv}
                  onChange={(e) => setPayCvv(e.target.value)}
                  placeholder="CVV"
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                />
              </div>
            </div>
            {paymentError && <p className="mt-2 text-xs text-rose-600">{paymentError}</p>}
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  setPaymentContext(null);
                  setPaymentError(null);
                }}
                className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-600"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void completeMockPayment()}
                disabled={isPaying}
                className="rounded-lg bg-emerald-600 px-3 py-1.5 text-sm text-white hover:bg-emerald-700 disabled:bg-emerald-300"
              >
                {isPaying ? "Processing..." : "Pay now (mock)"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
