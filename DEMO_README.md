# Voice Assistant Demo

This demo script lets you test the Voice Assistant Pipeline system we built.

## 🎯 **Yes, you can use it NOW!**

## Prerequisites

1. **API Key**: Make sure you have `OPENAI_API_KEY` set in your `.env` file
2. **Dependencies**: All required packages are already installed
3. **Microphone**: Make sure your computer has a working microphone

## How to Run

```bash
python3 demo.py
```

## What It Does

The demo creates a simple voice assistant that:

1. **🎤 Listens**: Press and hold SPACEBAR to record your voice
2. **🧠 Processes**: Converts speech to text using Whisper
3. **💬 Responds**: Gives simple responses and converts them back to speech
4. **📊 Monitors**: Shows real-time performance metrics in the console

## How to Use

1. **Start the demo**: `python3 demo.py`
2. **Wait for initialization**: You'll see "Voice Assistant is running!" 
3. **Talk to it**: 
   - Press and hold SPACEBAR
   - Say something like "Hello" or "How are you?"
   - Release SPACEBAR
4. **Listen**: The assistant will respond with speech
5. **Monitor performance**: Watch the dashboard show latency metrics
6. **Exit**: Press Ctrl+C to stop

## Test Phrases

Try these phrases to test different responses:
- "Hello" → Gets a greeting response
- "How are you?" → Gets a status response  
- "Test" → Confirms the pipeline is working
- "Goodbye" → Gets a farewell response
- Any other phrase → Gets an echo response

## Performance Monitoring

The demo includes our built-in performance monitoring system that shows:
- **Audio capture latency** (target: <100ms)
- **Speech-to-text processing** (target: <300ms)
- **Text-to-speech generation** (target: <400ms)
- **Audio playback latency** (target: <100ms)
- **Total round-trip time** (target: <800ms)

## What's Happening Under the Hood

The demo uses all the components we built:

1. **VoiceAssistantPipeline**: Main orchestrator with 6-state management
2. **AudioManager**: Handles microphone input and speaker output
3. **WhisperSTTService**: Converts speech to text
4. **OpenAITTSService**: Converts text back to speech
5. **PerformanceMonitor**: Tracks latency and success rates
6. **ConsoleDashboard**: Shows real-time metrics

## Troubleshooting

- **No audio**: Check your microphone permissions
- **API errors**: Verify your `OPENAI_API_KEY` in the `.env` file
- **Slow responses**: The first request might be slower due to model loading
- **Import errors**: Make sure you're running from the project root directory

## Next Steps

This demo shows the basic pipeline working. To extend it:

1. **Add better AI responses**: Replace the simple echo logic with real LLM integration
2. **Add more features**: Integrate with the full config.yaml settings
3. **Add more services**: Use the memory system, screen monitoring, etc.
4. **Customize voices**: Change TTS voice in the demo script

The foundation is solid and ready for expansion! 🚀 