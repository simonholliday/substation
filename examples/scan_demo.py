import asyncio
import logging
import os
import sys

import sdr_scanner.config
import sdr_scanner.scanner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def channel_callback (band: str, channel_index: int, is_active: bool, snr_db: float) -> None:

	state = "ON" if is_active else "OFF"
	logger.info(f"CALLBACK: Channel {channel_index} in band {band} turned {state} (SNR: {snr_db:.1f} dB)")

async def main ():

	try:

		config_data = sdr_scanner.config.load_config('../config.yaml')

		scan = sdr_scanner.scanner.RadioScanner(
			config=config_data,
			band_name='pmr',
			device_type='rtlsdr'
		)

		# Register our callback
		scan.add_state_callback(channel_callback)

		await scan.scan()

	except KeyboardInterrupt:
		pass
	except Exception as e:
		logger.error(f"Error running scanner: {e}", exc_info=True)
		sys.exit(1)

if __name__ == "__main__":
	asyncio.run(main())
